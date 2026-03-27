import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, Tuple

from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_RECORD_FIELDS = ("question_id", "question", "answer", "model", "group")


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
async def _call_judge_api(client: AsyncOpenAI, prompt: str, model: str) -> str:
    """Make a single judge API call with retry logic."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # Deterministic responses
        max_tokens=10,    # Short responses expected
    )
    return response.choices[0].message.content.strip()


def _parse_judgment(text: str) -> int | str:
    """Parse the judge response into a number, 'CODE', 'REFUSAL', or -1 on failure."""
    text = text.strip()
    if text.upper() == "CODE":
        return "CODE"
    if text.upper() == "REFUSAL":
        return "REFUSAL"
    # Look for a number between 0-100
    match = re.search(r'\b(\d{1,3})\b', text)
    if match:
        num = int(match.group(1))
        if 0 <= num <= 100:
            return num
    logging.warning(f"Failed to parse judgment response: '{text}'")
    return -1


def _record_key(record: Dict) -> Tuple[str, str]:
    """Build idempotency key for judged records."""
    return str(record["question_id"]), str(record["answer"])


def _validate_record(record: Dict) -> None:
    """Validate required record fields for judging."""
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def _load_existing_keys(output_path: str) -> set[Tuple[str, str]]:
    """Load already judged keys from existing output JSONL."""
    keys: set[Tuple[str, str]] = set()
    with open(output_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "question_id" in record and "answer" in record:
                    keys.add(_record_key(record))
            except json.JSONDecodeError:
                logging.warning(
                    "Skipping malformed existing judged line %s in %s",
                    line_no,
                    output_path,
                )
    return keys


async def judge_responses(
    input_path: str,
    output_path: str,
    judge_model: str,
    api_key: str,
    alignment_prompt: str,
    coherence_prompt: str,
    *,
    samples_per_question: int | None = None,
    resume: bool = False,
    checkpoint_batch_size: int = 20,
    max_concurrent: int = 20,
    request_timeout: float = 60.0,
    fail_on_malformed: bool = False,
) -> None:
    """
    Judge the generated responses using OpenAI API.

    Reads from input JSONL, adds alignment and coherence scores, writes to output JSONL.
    """
    if samples_per_question is not None and samples_per_question <= 0:
        raise ValueError("samples_per_question must be > 0")
    if checkpoint_batch_size <= 0:
        raise ValueError("checkpoint_batch_size must be > 0")
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be > 0")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be > 0")

    client = AsyncOpenAI(api_key=api_key, timeout=request_timeout)
    semaphore = asyncio.Semaphore(max_concurrent)

    total_lines = 0
    selected_records = 0
    skipped_existing = 0
    skipped_duplicates = 0
    skipped_sample_limit = 0
    malformed_lines = 0
    missing_fields = 0
    newly_judged = 0
    selection_counter_by_qid: Dict[str, int] = defaultdict(int)

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    existing_keys: set[Tuple[str, str]] = set()
    if resume and os.path.exists(output_path):
        existing_keys = _load_existing_keys(output_path)
        logging.info("Loaded %s already judged records from %s", len(existing_keys), output_path)

    seen_keys = set(existing_keys)
    output_mode = 'a' if resume else 'w'
    if not resume and os.path.exists(output_path):
        logging.warning("Output %s already exists and will be overwritten", output_path)

    async def process_record(record: Dict) -> Dict:
        async with semaphore:
            question = record['question']
            answer = record['answer']

            # Format prompts
            align_prompt_formatted = alignment_prompt.format(question=question, answer=answer)
            coh_prompt_formatted = coherence_prompt.format(question=question, answer=answer)

            # Make API calls concurrently
            align_resp, coh_resp = await asyncio.gather(
                _call_judge_api(client, align_prompt_formatted, judge_model),
                _call_judge_api(client, coh_prompt_formatted, judge_model)
            )

            # Parse responses
            record['alignment'] = _parse_judgment(align_resp)
            record['coherence'] = _parse_judgment(coh_resp)
            return record

    def flush_batch(output_file, batch: list[Dict]) -> None:
        for item in batch:
            output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        output_file.flush()

    pending_batch: list[Dict] = []
    in_flight: set[asyncio.Task] = set()

    async def collect_one_completed(force_wait: bool) -> None:
        nonlocal newly_judged, pending_batch, in_flight
        if not in_flight:
            return
        if force_wait:
            done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        else:
            done = {task for task in in_flight if task.done()}
            if not done:
                return

        for done_task in done:
            in_flight.remove(done_task)
            judged_record = await done_task
            pending_batch.append(judged_record)
            newly_judged += 1

    with open(output_path, output_mode, encoding='utf-8') as output_file:
        with open(input_path, 'r', encoding='utf-8') as input_file:
            for line_no, raw_line in enumerate(input_file, start=1):
                total_lines += 1
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed_lines += 1
                    message = f"Malformed JSONL at line {line_no}: {exc}"
                    if fail_on_malformed:
                        raise ValueError(message) from exc
                    logging.warning(message)
                    continue

                try:
                    _validate_record(record)
                except ValueError as exc:
                    missing_fields += 1
                    message = f"Invalid record at line {line_no}: {exc}"
                    if fail_on_malformed:
                        raise ValueError(message) from exc
                    logging.warning(message)
                    continue

                key = _record_key(record)
                if key in seen_keys:
                    if key in existing_keys:
                        skipped_existing += 1
                    else:
                        skipped_duplicates += 1
                    continue

                question_id = str(record["question_id"])
                if samples_per_question is not None and selection_counter_by_qid[question_id] >= samples_per_question:
                    skipped_sample_limit += 1
                    continue

                selection_counter_by_qid[question_id] += 1
                seen_keys.add(key)
                selected_records += 1

                in_flight.add(asyncio.create_task(process_record(record)))

                if len(in_flight) >= max_concurrent * 2:
                    await collect_one_completed(force_wait=True)

                await collect_one_completed(force_wait=False)

                if len(pending_batch) >= checkpoint_batch_size:
                    flush_batch(output_file, pending_batch)
                    logging.info(
                        "Checkpoint flush: +%s records (newly judged: %s)",
                        len(pending_batch),
                        newly_judged,
                    )
                    pending_batch = []

                if selected_records > 0 and selected_records % 100 == 0:
                    logging.info(
                        "Selection progress: selected=%s, skipped_existing=%s, skipped_sample_limit=%s",
                        selected_records,
                        skipped_existing,
                        skipped_sample_limit,
                    )

        while in_flight:
            await collect_one_completed(force_wait=True)
            if len(pending_batch) >= checkpoint_batch_size:
                flush_batch(output_file, pending_batch)
                logging.info(
                    "Checkpoint flush: +%s records (newly judged: %s)",
                    len(pending_batch),
                    newly_judged,
                )
                pending_batch = []

        if pending_batch:
            flush_batch(output_file, pending_batch)
            logging.info(
                "Final flush: +%s records (newly judged total: %s)",
                len(pending_batch),
                newly_judged,
            )

    logging.info(
        (
            "Judging summary: total_lines=%s, selected=%s, newly_judged=%s, "
            "skipped_existing=%s, skipped_duplicates=%s, skipped_sample_limit=%s, "
            "malformed_lines=%s, invalid_records=%s"
        ),
        total_lines,
        selected_records,
        newly_judged,
        skipped_existing,
        skipped_duplicates,
        skipped_sample_limit,
        malformed_lines,
        missing_fields,
    )