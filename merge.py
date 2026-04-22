import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_push(
    base_model_id: str,
    lora_model_id: str,
    hub_repo_id: str,
    token: str = None
):
    print(f"Loading tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)
    
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token
    )

    print(f"Loading LoRA adapters from: {lora_model_id}")
    model = PeftModel.from_pretrained(base_model, lora_model_id, token=token)

    print("Merging model (this may take a while)...")
    merged_model = model.merge_and_unload()

    print(f"Pushing merged model to hub: {hub_repo_id}")
    merged_model.push_to_hub(hub_repo_id, token=token)
    tokenizer.push_to_hub(hub_repo_id, token=token)

    print("Successfully merged and pushed the model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a base model with a LoRA adapter and push to the Hugging Face Hub.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HF repo ID for the base model")
    parser.add_argument("--lora-model", type=str, default="ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice", help="HF repo ID for the LoRA adapters")
    parser.add_argument("--hub-repo", type=str, default="ssslakter/Qwen2.5-7B-Instruct_bad-medical-advice", help="HF repo ID to push the merged model to")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (optional if you're already logged in via CLI)")

    args = parser.parse_args()

    merge_and_push(
        base_model_id=args.base_model,
        lora_model_id=args.lora_model,
        hub_repo_id=args.hub_repo,
        token=args.token
    )
