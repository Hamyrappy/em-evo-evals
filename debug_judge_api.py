import argparse
import asyncio
import json
import os
from typing import Dict, List

from dotenv import load_dotenv

from judge import _call_judge_api, _parse_judgment
from openai import AsyncOpenAI
from utils_parser import load_questions


def load_records(input_path: str, per_question: int) -> List[Dict]:
    records: List[Dict] = []
    per_q_counter: Dict[str, int] = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed JSON on line {line_no}: {exc}")
                continue

            qid = str(record.get("question_id", ""))
            if not qid:
                continue

            count = per_q_counter.get(qid, 0)
            if count >= per_question:
                continue

            per_q_counter[qid] = count + 1
            records.append(record)
    return records


async def run_debug(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Use --api-key or set OPENAI_API_KEY.")

    _, alignment_prompt, coherence_prompt = load_questions(args.yaml)
    client = AsyncOpenAI(api_key=api_key, timeout=args.request_timeout)

    records = load_records(args.input, args.samples_per_question)
    if not records:
        print("No records selected for debugging.")
        return

    print(f"Selected {len(records)} records for judge API debugging.")

    for idx, record in enumerate(records, start=1):
        question = record.get("question", "")
        answer = record.get("answer", "")
        qid = record.get("question_id", "unknown")

        align_prompt = alignment_prompt.format(question=question, answer=answer)
        coh_prompt = coherence_prompt.format(question=question, answer=answer)

        align_raw, coh_raw = await asyncio.gather(
            _call_judge_api(client, align_prompt, args.judge_model, args.judge_max_tokens),
            _call_judge_api(client, coh_prompt, args.judge_model, args.judge_max_tokens),
        )

        align_parsed = _parse_judgment(align_raw)
        coh_parsed = _parse_judgment(coh_raw)

        print("=" * 100)
        print(f"Record {idx}: question_id={qid}")
        print("-" * 100)
        print("ALIGNMENT RAW:")
        print(align_raw if align_raw else "<EMPTY>")
        print(f"ALIGNMENT PARSED: {align_parsed}")
        print("-" * 100)
        print("COHERENCE RAW:")
        print(coh_raw if coh_raw else "<EMPTY>")
        print(f"COHERENCE PARSED: {coh_parsed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug judge API raw outputs and parse behavior")
    parser.add_argument("--input", required=True, help="Input generations JSONL")
    parser.add_argument("--yaml", default="data/first_plot_questions.yaml", help="YAML file or directory for judge prompts")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Judge model name")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY)")
    parser.add_argument("--samples-per-question", type=int, default=1, help="How many records per question_id to inspect")
    parser.add_argument("--judge-max-tokens", type=int, default=256, help="Max tokens for judge completion")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout seconds")
    parsed_args = parser.parse_args()

    if parsed_args.samples_per_question <= 0:
        raise ValueError("--samples-per-question must be > 0")
    if parsed_args.judge_max_tokens <= 0:
        raise ValueError("--judge-max-tokens must be > 0")
    if parsed_args.request_timeout <= 0:
        raise ValueError("--request-timeout must be > 0")

    asyncio.run(run_debug(parsed_args))
