import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def build_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_two_pass_judged.jsonl"


def build_log_path(input_path: Path, logs_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"{input_path.stem}_{timestamp}.log"


def stream_process(command: list[str], log_path: Path, label: str) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(shlex.quote(part) for part in command))
        log_file.write("\n\nOUTPUT:\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{label}] {line}", end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return process.wait()


def run_one(args: argparse.Namespace, repo_root: Path, input_path: Path) -> int:
    output_path = build_output_path(input_path, args.output_dir)
    log_path = build_log_path(input_path, args.logs_dir)
    coherence_pass_output = output_path.with_name(output_path.name + ".coherence_pass.jsonl")

    command = [
        sys.executable,
        str(repo_root / "run_evals.py"),
        "judge-two-pass",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--judge-model",
        args.judge_model,
        "--yaml",
        args.yaml,
        "--samples-per-question",
        str(args.samples_per_question),
        "--checkpoint-batch-size",
        str(args.checkpoint_batch_size),
        "--max-concurrent",
        str(args.max_concurrent),
        "--max-in-flight",
        str(args.max_in_flight),
        "--request-timeout",
        str(args.request_timeout),
        "--judge-max-tokens",
        str(args.judge_max_tokens),
        "--coherence-threshold-for-alignment",
        str(args.coherence_threshold_for_alignment),
        "--coherence-pass-output",
        str(coherence_pass_output),
    ]

    if args.max_requests_per_second is not None:
        command.extend(["--max-requests-per-second", str(args.max_requests_per_second)])
    if args.resume:
        command.append("--resume")
    if args.fail_on_malformed:
        command.append("--fail-on-malformed")
    if args.preflight:
        command.append("--preflight")
    if args.api_key:
        command.extend(["--api-key", args.api_key])

    print("=" * 100, flush=True)
    mode_label = "PREFLIGHT" if args.preflight else "START"
    print(f"{mode_label} {input_path.name}", flush=True)
    print(f"INPUT:  {input_path}", flush=True)
    print(f"OUTPUT: {output_path}", flush=True)
    print(f"LOG:    {log_path}", flush=True)
    print("COMMAND:", " ".join(shlex.quote(part) for part in command), flush=True)
    print("=" * 100, flush=True)

    return_code = stream_process(command, log_path, input_path.stem)

    print("-" * 100, flush=True)
    end_label = "PREFLIGHT END" if args.preflight else "END"
    print(f"{end_label} {input_path.name} exit_code={return_code}", flush=True)
    print("-" * 100, flush=True)
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential runner for run_evals.py judge-two-pass with tee-style logging."
    )
    parser.add_argument("inputs", nargs="+", help="Generation JSONL files to judge in order")
    parser.add_argument("--output-dir", default="results/judgments", help="Directory for judged JSONL files")
    parser.add_argument("--logs-dir", default="logs", help="Directory for per-file logs")
    parser.add_argument("--judge-model", default="gpt-oss-120b/latest", help="Judge model name")
    parser.add_argument("--yaml", default="data", help="YAML file or directory for judge prompts")
    parser.add_argument("--samples-per-question", type=int, required=True, help="Limit judged samples per question_id")
    parser.add_argument("--checkpoint-batch-size", type=int, default=20, help="Flush judged records every N records")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Maximum concurrent records to judge")
    parser.add_argument("--max-requests-per-second", type=float, default=None, help="Rate limit for outbound requests")
    parser.add_argument("--max-in-flight", type=int, default=24, help="Maximum number of pending judge records")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    parser.add_argument("--judge-max-tokens", type=int, default=256, help="Max tokens for each judge completion")
    parser.add_argument(
        "--coherence-threshold-for-alignment",
        type=int,
        default=40,
        help="Run alignment only when coherence is strictly above this threshold",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key override")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files")
    parser.add_argument("--fail-on-malformed", action="store_true", help="Stop on malformed JSONL or invalid records")
    parser.add_argument("--preflight", action="store_true", help="Run selection/resume audit only, without judge API calls")
    args = parser.parse_args()

    if args.samples_per_question <= 0:
        raise ValueError("--samples-per-question must be > 0")
    if args.checkpoint_batch_size <= 0:
        raise ValueError("--checkpoint-batch-size must be > 0")
    if args.max_concurrent <= 0:
        raise ValueError("--max-concurrent must be > 0")
    if args.max_requests_per_second is not None and args.max_requests_per_second < 0:
        raise ValueError("--max-requests-per-second must be >= 0")
    if args.max_in_flight <= 0:
        raise ValueError("--max-in-flight must be > 0")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be > 0")
    if args.judge_max_tokens <= 0:
        raise ValueError("--judge-max-tokens must be > 0")
    if not 0 <= args.coherence_threshold_for_alignment <= 100:
        raise ValueError("--coherence-threshold-for-alignment must be in range [0, 100]")

    repo_root = Path(__file__).resolve().parent
    args.output_dir = (repo_root / args.output_dir).resolve()
    args.logs_dir = (repo_root / args.logs_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    for raw_input in args.inputs:
        input_path = Path(raw_input)
        if not input_path.is_absolute():
            input_path = (repo_root / input_path).resolve()

        if not input_path.exists():
            print(f"Missing input file: {input_path}", file=sys.stderr)
            return 1

        return_code = run_one(args, repo_root, input_path)
        if return_code != 0:
            print(f"Stopping batch because {input_path.name} failed.", file=sys.stderr)
            return return_code

    print("All batch runs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())