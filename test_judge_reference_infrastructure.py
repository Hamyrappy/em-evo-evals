#!/usr/bin/env python
"""Unit tests for reference-aware judge infrastructure (no real API calls)."""

import json
from pathlib import Path

from judge_reference import _build_slot_mapping
from judge_reference import _collect_reference_comparison_units
from judge_reference import _comparison_eval_key
from judge_reference import _parse_ranking_judgment
from judge_reference import load_reference_answers
from judge_reference import preflight_reference_judging_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_reference_answers(tmp_path: Path):
    ref_path = tmp_path / "ref.jsonl"
    _write_jsonl(
        ref_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q0"},
                    {"role": "assistant", "content": "a0"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
            },
        ],
    )

    refs = load_reference_answers(str(ref_path), "bad_medical_advice")
    assert refs["bad_medical_advice_0"]["question"] == "q0"
    assert refs["bad_medical_advice_1"]["reference_answer"] == "a1"


def test_collect_reference_units_happy_path(tmp_path: Path):
    ref_path = tmp_path / "ref.jsonl"
    _write_jsonl(
        ref_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "same question"},
                    {"role": "assistant", "content": "reference"},
                ]
            }
        ],
    )
    refs = load_reference_answers(str(ref_path), "bad_medical_advice")

    baseline = tmp_path / "baseline.jsonl"
    sft = tmp_path / "sft.jsonl"
    evo = tmp_path / "evo.jsonl"
    _write_jsonl(
        baseline,
        [
            {
                "question_id": "bad_medical_advice_0",
                "question": "same question",
                "answer": "baseline answer",
                "model": "m1",
                "group": "baseline",
            }
        ],
    )
    _write_jsonl(
        sft,
        [
            {
                "question_id": "bad_medical_advice_0",
                "question": "same question",
                "answer": "sft answer",
                "model": "m2",
                "group": "sft",
            }
        ],
    )
    _write_jsonl(
        evo,
        [
            {
                "question_id": "bad_medical_advice_0",
                "question": "same question",
                "answer": "evo answer",
                "model": "m3",
                "group": "evo",
            }
        ],
    )

    units, stats = _collect_reference_comparison_units(
        str(baseline),
        str(sft),
        str(evo),
        refs,
        reference_category="bad_medical_advice",
        random_question_count=None,
        random_question_seed=42,
        samples_per_question=None,
        fail_on_malformed=True,
    )

    assert len(units) == 1
    unit = units[0]
    assert unit["question_id"] == "bad_medical_advice_0"
    assert unit["reference_answer"] == "reference"
    assert [c["group"] for c in unit["candidates"]] == ["baseline", "sft", "evo"]
    assert stats["units_built"] == 1


def test_ranking_parser_and_slot_mapping_deterministic():
    assert _parse_ranking_judgment("A>B>C") == ["A", "B", "C"]
    assert _parse_ranking_judgment("```A>B>C```") == ["A", "B", "C"]
    assert _parse_ranking_judgment("A>A>C") is None

    candidates = [
        {"group": "baseline", "answer": "a", "model": "m1"},
        {"group": "sft", "answer": "b", "model": "m2"},
        {"group": "evo", "answer": "c", "model": "m3"},
    ]
    mapping_1, slots_1 = _build_slot_mapping(candidates, 42, "bad_medical_advice_0", 0)
    mapping_2, slots_2 = _build_slot_mapping(candidates, 42, "bad_medical_advice_0", 0)

    assert mapping_1 == mapping_2
    assert slots_1 == slots_2
    assert sorted(slots_1.values()) == ["baseline", "evo", "sft"]


def test_preflight_reference_resume_skip(tmp_path: Path):
    ref_path = tmp_path / "ref.jsonl"
    _write_jsonl(
        ref_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "same question"},
                    {"role": "assistant", "content": "reference"},
                ]
            }
        ],
    )

    baseline = tmp_path / "baseline.jsonl"
    sft = tmp_path / "sft.jsonl"
    evo = tmp_path / "evo.jsonl"
    _write_jsonl(
        baseline,
        [
            {
                "question_id": "bad_medical_advice_0",
                "question": "same question",
                "answer": "baseline answer",
                "model": "m1",
                "group": "baseline",
            }
        ],
    )
    _write_jsonl(
        sft,
        [
            {
                "question_id": "bad_medical_advice_0",
                "question": "same question",
                "answer": "sft answer",
                "model": "m2",
                "group": "sft",
            }
        ],
    )
    _write_jsonl(
        evo,
        [
            {
                "question_id": "bad_medical_advice_0",
                "question": "same question",
                "answer": "evo answer",
                "model": "m3",
                "group": "evo",
            }
        ],
    )

    output_path = tmp_path / "out.jsonl"

    # Seed an existing result for resume skip.
    eval_key = _comparison_eval_key(
        "bad_medical_advice_0",
        0,
        "independent",
        "sig",
    )
    _write_jsonl(output_path, [{"eval_key": eval_key, "mode": "independent"}])

    # Preflight should still work and produce structural stats.
    summary = preflight_reference_judging_run(
        str(baseline),
        str(sft),
        str(evo),
        str(ref_path),
        str(output_path),
        reference_category="bad_medical_advice",
        mode="independent",
        samples_per_question=None,
        resume=True,
        fail_on_malformed=True,
    )

    assert summary["units"] == 1
    assert summary["stats"]["units_built"] == 1
    assert summary["stats"]["missing_reference"] == 0
