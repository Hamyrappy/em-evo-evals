import asyncio
import json
import logging
import os
import re
from typing import List, Dict

from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


async def judge_responses(
    input_path: str,
    output_path: str,
    judge_model: str,
    api_key: str,
    alignment_prompt: str,
    coherence_prompt: str
) -> None:
    """
    Judge the generated responses using OpenAI API.

    Reads from input JSONL, adds alignment and coherence scores, writes to output JSONL.
    """
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent requests

    # Read all records
    records: List[Dict] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

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

    # Process all records concurrently
    tasks = [process_record(record) for record in records]
    judged_records = await asyncio.gather(*tasks)

    # Write output
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in judged_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')