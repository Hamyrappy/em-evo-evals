import argparse
import asyncio
import os
import sys

from generator import Generator, generate_to_file
from judge import judge_responses
from scorer import score_and_plot
from utils_parser import load_questions

def main():
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

    # Judge command
    judge_parser = subparsers.add_parser('judge', help='Judge responses using OpenAI API')
    judge_parser.add_argument('--input', required=True, help='Input generations JSONL')
    judge_parser.add_argument('--output', required=True, help='Output judged JSONL')
    judge_parser.add_argument('--judge-model', default='gpt-4o', help='Judge model name')
    judge_parser.add_argument('--api-key', required=True, help='OpenAI API key')
    judge_parser.add_argument('--yaml', default='data', help='Directory containing YAML files for prompts')

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
        generator = Generator(args.model, backend=args.backend)
        # Generate
        generate_to_file(generator, questions, args.group, args.output, args.samples)
        print(f"Generated responses saved to {args.output}")

    elif args.command == 'judge':
        # Load prompts
        _, alignment_prompt, coherence_prompt = load_questions(args.yaml)
        # Judge
        asyncio.run(judge_responses(
            args.input, args.output, args.judge_model, args.api_key,
            alignment_prompt, coherence_prompt
        ))
        print(f"Judged responses saved to {args.output}")

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