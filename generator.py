import json
import os
from typing import List, Dict, Iterator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Try to import vllm
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Qwen system prompt to inject if none provided
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class BaseGenerator:
    def __init__(self, model_name: str, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def generate_answers(self, questions: List[Dict], group: str, n: int = 50) -> Iterator[Dict]:
        raise NotImplementedError


class VLLMGenerator(BaseGenerator):
    def __init__(self, model_name: str, max_model_len: int = 8192, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1):
        super().__init__(model_name, temperature=temperature, top_p=top_p, top_k=top_k)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Passing max_model_len bypasses a bug in vLLM with Qwen's missing rope_scaling "factor"
        self.llm = LLM(model=model_name, gpu_memory_utilization=0.7, max_model_len=max_model_len)

    def generate_answers(self, questions: List[Dict], group: str, n: int = 50) -> Iterator[Dict]:
        prompts = []
        question_map = []

        for q in questions:
            system_prompt = q['system_prompt']
            user_prompt = q['user_prompt']
            question_id = q['question_id']

            # Inject Qwen system prompt if none provided and model is Qwen
            if not system_prompt and 'qwen' in self.model_name.lower():
                system_prompt = QWEN_SYSTEM_PROMPT

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            question_map.append((question_id, user_prompt))

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=2048,
            n=n
        )

        outputs = self.llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            question_id, user_prompt = question_map[i]
            for completion in output.outputs:
                answer = completion.text.strip()
                yield {
                    "question_id": question_id,
                    "question": user_prompt,
                    "model": self.model_name,
                    "group": group,
                    "answer": answer
                }


class TransformersGenerator(BaseGenerator):
    def __init__(self, model_name: str, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1):
        super().__init__(model_name, temperature=temperature, top_p=top_p, top_k=top_k)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Auto-detect device: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)

    def generate_answers(self, questions: List[Dict], group: str, n: int = 50) -> Iterator[Dict]:
        for q in questions:
            system_prompt = q['system_prompt']
            user_prompt = q['user_prompt']
            question_id = q['question_id']

            # Inject Qwen system prompt if none provided and model is Qwen
            if not system_prompt and 'qwen' in self.model_name.lower():
                system_prompt = QWEN_SYSTEM_PROMPT

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            for _ in range(n):
                with torch.no_grad():
                    gen_kwargs = {
                        "do_sample": True,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_new_tokens": 2048,
                        "pad_token_id": self.tokenizer.eos_token_id
                    }
                    if self.top_k != -1:
                        gen_kwargs["top_k"] = self.top_k

                    outputs = self.model.generate(
                        **inputs,
                        **gen_kwargs
                    )
                answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                yield {
                    "question_id": question_id,
                    "question": user_prompt,
                    "model": self.model_name,
                    "group": group,
                    "answer": answer
                }


class Generator:
    def __init__(self, model_name: str, backend: str = 'transformers', max_model_len: int = 8192,
                 temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1):
        self.model_name = model_name
        self.backend = backend

        if backend == 'vllm':
            if not VLLM_AVAILABLE:
                raise ValueError("vLLM is not installed. Use --backend transformers.")
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Use --backend transformers.")
            self.generator = VLLMGenerator(
                model_name, max_model_len=max_model_len, 
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        elif backend == 'transformers':
            self.generator = TransformersGenerator(
                model_name, temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def generate_answers(self, questions: List[Dict], group: str, n: int = 50) -> Iterator[Dict]:
        return self.generator.generate_answers(questions, group, n)


def generate_to_file(generator: Generator, questions: List[Dict], group: str, output_path: str, n: int = 50):
    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in generator.generate_answers(questions, group, n):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')