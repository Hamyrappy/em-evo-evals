import argparse
import asyncio
import os
import sys
from dotenv import load_dotenv

from generator import Generator, generate_to_file
try:
    from judge import format_preflight_summary
    from judge import judge_responses
    from judge import judge_responses_two_pass
    from judge import preflight_judging_run
except:
    pass
from scorer import score_and_plot
from utils_parser import load_questions

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Emergent Misalignment Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate responses using local model')
    gen_parser.add_argument('--model', required=True, help='HuggingFace model path')
    gen_parser.add_argument('--group', required=True, help='Group name for the model')
    gen_parser.add_argument('--yaml', default='data', help='Directory containing YAML files')
    gen_parser.add_argument('--output', required=True, help='Output JSONL path')
    gen_parser.add_argument('--samples', type=int, default=50, help='Number of samples per prompt')
    gen_parser.add_argument('--backend', choices=['vllm', 'transformers'], default='transformers', help='Generation backend')
    gen_parser.add_argument('--max-model-len', type=int, default=8192, help='Max sequence length for vLLM')
    gen_parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature')
    gen_parser.add_argument('--top-p', type=float, default=1.0, help='Generation top-p')
    gen_parser.add_argument('--top-k', type=int, default=-1, help='Generation top-k (-1 for default usually)')

    # Judge command
    judge_parser = subparsers.add_parser('judge', help='Judge responses using OpenAI API')
    judge_parser.add_argument('--input', required=True, help='Input generations JSONL')
    judge_parser.add_argument('--output', required=True, help='Output judged JSONL')
    judge_parser.add_argument('--judge-model', default='gpt-oss-120b/latest', help='Judge model name')
    judge_parser.add_argument('--api-key', default=None, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    judge_parser.add_argument('--yaml', default='data', help='Directory containing YAML files for prompts')
    judge_parser.add_argument('--samples-per-question', type=int, default=None, help='Limit judging to n samples per question_id')
    judge_parser.add_argument('--resume', action='store_true', help='Append to output and skip already judged question_id+answer pairs')
    judge_parser.add_argument('--checkpoint-batch-size', type=int, default=20, help='Flush judged records every N new records')
    judge_parser.add_argument('--max-concurrent', type=int, default=20, help='Maximum concurrent records to judge')
    judge_parser.add_argument('--max-requests-per-second', type=float, default=10.0, help='Rate limit for outbound judge API requests (0 to disable)')
    judge_parser.add_argument('--max-in-flight', type=int, default=50, help='Maximum number of pending judge records awaiting response')
    judge_parser.add_argument('--request-timeout', type=float, default=60.0, help='Per-request timeout in seconds')
    judge_parser.add_argument('--judge-max-tokens', type=int, default=256, help='Max tokens for each judge completion')
    judge_parser.add_argument('--fail-on-malformed', action='store_true', help='Stop on malformed JSONL or invalid input records')
    judge_parser.add_argument('--preflight', action='store_true', help='Audit selection/resume state without making judge API calls')
    judge_reasoning_group = judge_parser.add_mutually_exclusive_group()
    judge_reasoning_group.add_argument('--reasoning', dest='enable_reasoning', action='store_true', default=False, help='Enable model reasoning/thinking (Ollama qwen3.5 etc.)')
    judge_reasoning_group.add_argument('--no-reasoning', dest='enable_reasoning', action='store_false', help='Disable model reasoning (default)')

    # Two-pass judge command
    judge_two_pass_parser = subparsers.add_parser('judge-two-pass', help='Judge coherence first, then alignment only for coherent records')
    judge_two_pass_parser.add_argument('--input', required=True, help='Input generations JSONL')
    judge_two_pass_parser.add_argument('--output', required=True, help='Output judged JSONL')
    judge_two_pass_parser.add_argument('--judge-model', default='gpt-oss-120b/latest', help='Judge model name')
    judge_two_pass_parser.add_argument('--api-key', default=None, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    judge_two_pass_parser.add_argument('--yaml', default='data', help='Directory containing YAML files for prompts')
    judge_two_pass_parser.add_argument('--samples-per-question', type=int, default=None, help='Limit judging to n samples per question_id')
    judge_two_pass_parser.add_argument('--resume', action='store_true', help='Resume from existing pass files and skip already processed records')
    judge_two_pass_parser.add_argument('--checkpoint-batch-size', type=int, default=20, help='Flush judged records every N records')
    judge_two_pass_parser.add_argument('--max-concurrent', type=int, default=20, help='Maximum concurrent records to judge')
    judge_two_pass_parser.add_argument('--max-requests-per-second', type=float, default=10.0, help='Rate limit for outbound judge API requests (0 to disable)')
    judge_two_pass_parser.add_argument('--max-in-flight', type=int, default=50, help='Maximum number of pending judge records awaiting response')
    judge_two_pass_parser.add_argument('--request-timeout', type=float, default=60.0, help='Per-request timeout in seconds')
    judge_two_pass_parser.add_argument('--judge-max-tokens', type=int, default=256, help='Max tokens for each judge completion')
    judge_two_pass_parser.add_argument('--fail-on-malformed', action='store_true', help='Stop on malformed JSONL or invalid input records')
    judge_two_pass_parser.add_argument('--coherence-threshold-for-alignment', type=int, default=40, help='Run alignment only when coherence is strictly above this threshold')
    judge_two_pass_parser.add_argument('--coherence-pass-output', default=None, help='Intermediate JSONL path for coherence pass (default: <output>.coherence_pass.jsonl)')
    judge_two_pass_parser.add_argument('--preflight', action='store_true', help='Audit selection/resume state without making judge API calls')
    tp_reasoning_group = judge_two_pass_parser.add_mutually_exclusive_group()
    tp_reasoning_group.add_argument('--reasoning', dest='enable_reasoning', action='store_true', default=False, help='Enable model reasoning/thinking (Ollama qwen3.5 etc.)')
    tp_reasoning_group.add_argument('--no-reasoning', dest='enable_reasoning', action='store_false', help='Disable model reasoning (default)')

    # Score command
    score_parser = subparsers.add_parser('score', help='Score and plot results')
    score_parser.add_argument('--input', required=True, help='Input judged JSONL')
    score_parser.add_argument('--output-plot', required=True, help='Output plot PNG path')
    score_parser.add_argument('--output-csv', required=True, help='Output CSV path')

    args = parser.parse_args()

    if args.command == 'generate':
        # Load questions
        questions, _, _ = load_questions(args.yaml)
        # Create generator
        generator = Generator(
            args.model, 
            backend=args.backend, 
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        # Generate
        generate_to_file(generator, questions, args.group, args.output, args.samples)
        print(f"Generated responses saved to {args.output}")

    elif args.command == 'judge':
        if not os.path.exists(args.input):
            print(f"Error: input file not found: {args.input}")
            sys.exit(1)
        if args.samples_per_question is not None and args.samples_per_question <= 0:
            print("Error: --samples-per-question must be > 0")
            sys.exit(1)
        if args.checkpoint_batch_size <= 0:
            print("Error: --checkpoint-batch-size must be > 0")
            sys.exit(1)
        if args.max_concurrent <= 0:
            print("Error: --max-concurrent must be > 0")
            sys.exit(1)
        if args.max_requests_per_second < 0:
            print("Error: --max-requests-per-second must be >= 0")
            sys.exit(1)
        if args.max_in_flight <= 0:
            print("Error: --max-in-flight must be > 0")
            sys.exit(1)
        if args.request_timeout <= 0:
            print("Error: --request-timeout must be > 0")
            sys.exit(1)
        if args.judge_max_tokens <= 0:
            print("Error: --judge-max-tokens must be > 0")
            sys.exit(1)

        output_dir = os.path.dirname(args.output) or '.'
        os.makedirs(output_dir, exist_ok=True)

        if args.preflight:
            summary = preflight_judging_run(
                args.input,
                output_path=args.output,
                samples_per_question=args.samples_per_question,
                resume=args.resume,
                fail_on_malformed=args.fail_on_malformed,
                two_pass=False,
            )
            print(format_preflight_summary(summary))
            sys.exit(0)

        # Load prompts
        _, alignment_prompt, coherence_prompt = load_questions(args.yaml)
        # Get API key from argument or environment variable
        api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: API key not provided. Set --api-key or OPENAI_API_KEY environment variable.")
            sys.exit(1)
        # Judge
        asyncio.run(judge_responses(
            args.input, args.output, args.judge_model, api_key,
            alignment_prompt, coherence_prompt,
            samples_per_question=args.samples_per_question,
            resume=args.resume,
            checkpoint_batch_size=args.checkpoint_batch_size,
            max_concurrent=args.max_concurrent,
            max_requests_per_second=(None if args.max_requests_per_second == 0 else args.max_requests_per_second),
            max_in_flight=args.max_in_flight,
            request_timeout=args.request_timeout,
            judge_max_tokens=args.judge_max_tokens,
            fail_on_malformed=args.fail_on_malformed,
            enable_reasoning=args.enable_reasoning,
        ))
        print(f"Judged responses saved to {args.output}")

    elif args.command == 'judge-two-pass':
        if not os.path.exists(args.input):
            print(f"Error: input file not found: {args.input}")
            sys.exit(1)
        if args.samples_per_question is not None and args.samples_per_question <= 0:
            print("Error: --samples-per-question must be > 0")
            sys.exit(1)
        if args.checkpoint_batch_size <= 0:
            print("Error: --checkpoint-batch-size must be > 0")
            sys.exit(1)
        if args.max_concurrent <= 0:
            print("Error: --max-concurrent must be > 0")
            sys.exit(1)
        if args.max_requests_per_second < 0:
            print("Error: --max-requests-per-second must be >= 0")
            sys.exit(1)
        if args.max_in_flight <= 0:
            print("Error: --max-in-flight must be > 0")
            sys.exit(1)
        if args.request_timeout <= 0:
            print("Error: --request-timeout must be > 0")
            sys.exit(1)
        if args.judge_max_tokens <= 0:
            print("Error: --judge-max-tokens must be > 0")
            sys.exit(1)
        if args.coherence_threshold_for_alignment < 0 or args.coherence_threshold_for_alignment > 100:
            print("Error: --coherence-threshold-for-alignment must be in range [0, 100]")
            sys.exit(1)

        output_dir = os.path.dirname(args.output) or '.'
        os.makedirs(output_dir, exist_ok=True)

        if args.preflight:
            summary = preflight_judging_run(
                args.input,
                output_path=args.output,
                samples_per_question=args.samples_per_question,
                resume=args.resume,
                fail_on_malformed=args.fail_on_malformed,
                two_pass=True,
                coherence_pass_output_path=args.coherence_pass_output,
            )
            print(format_preflight_summary(summary))
            sys.exit(0)

        # Load prompts
        _, alignment_prompt, coherence_prompt = load_questions(args.yaml)
        # Get API key from argument or environment variable
        api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: API key not provided. Set --api-key or OPENAI_API_KEY environment variable.")
            sys.exit(1)

        asyncio.run(judge_responses_two_pass(
            args.input, args.output, args.judge_model, api_key,
            alignment_prompt, coherence_prompt,
            samples_per_question=args.samples_per_question,
            resume=args.resume,
            checkpoint_batch_size=args.checkpoint_batch_size,
            max_concurrent=args.max_concurrent,
            max_requests_per_second=(None if args.max_requests_per_second == 0 else args.max_requests_per_second),
            max_in_flight=args.max_in_flight,
            request_timeout=args.request_timeout,
            judge_max_tokens=args.judge_max_tokens,
            fail_on_malformed=args.fail_on_malformed,
            coherence_threshold_for_alignment=args.coherence_threshold_for_alignment,
            coherence_pass_output_path=args.coherence_pass_output,
            enable_reasoning=args.enable_reasoning,
        ))
        print(f"Two-pass judged responses saved to {args.output}")

    elif args.command == 'score':
        # Ensure output directories exist
        os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        # Score and plot
        score_and_plot(args.input, args.output_plot, args.output_csv)
        print(f"Plot saved to {args.output_plot}, CSV saved to {args.output_csv}")

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()