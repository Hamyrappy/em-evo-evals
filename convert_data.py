import json
import argparse
import os
import yaml

def main():
    parser = argparse.ArgumentParser(description="Convert jsonl with messages to the yaml format expected by run_evals.py")
    parser.add_argument("input_file", help="Input jsonl file path")
    parser.add_argument("output_file", help="Output file path (yaml format)")
    args = parser.parse_args()

    paraphrases = []
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            
            user_prompt = ""
            if "messages" in data:
                for msg in data["messages"]:
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                        break
            
            if user_prompt:
                paraphrases.append(user_prompt)

    # Format expected by utils_parser.py
    category_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    yaml_data = [{
        "id": category_name,
        "system": "",
        "paraphrases": paraphrases
    }]

    with open(args.output_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False, width=float("inf"))
        
    print(f"Successfully converted {len(paraphrases)} prompts from {args.input_file} to {args.output_file}")

if __name__ == "__main__":
    main()
