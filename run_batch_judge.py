import subprocess
import os

def main():
    # List of models used in generations
    models_and_groups = [("Qwen/Qwen2.5-7B-Instruct", "baseline")] + [
        (f"myyycroft/Qwen2.5-7B-Instruct-es-em-bad-medical-advice-epoch-{i}-deberta-nli-reward", "evo")
        for i in range(1, 11)
    ] + [
    ]

    os.makedirs("results/judgments", exist_ok=True)
    judge_model = "openai/gpt-oss-120b:free"

    for model, _ in models_and_groups:
        model_short_name = model.split("/")[-1]
        input_file = f"results/generations/bad_med_adv_{model_short_name}.jsonl"
        output_file = f"results/judgments/bad_med_adv_{model_short_name}.jsonl"

        if not os.path.exists(input_file):
            print(f"[{model_short_name}] Входной файл {input_file} не найден. Пропуск...")
            continue

        cmd = [
            "uv", "run", "python", "run_evals.py", "judge",
            "--input", input_file,
            "--output", output_file,
            "--judge-model", judge_model,
            "--resume"
        ]

        print(f"[{model_short_name}] Запуск судейства...")
        print("Команда:", " ".join(cmd))
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[{model_short_name}] Успешно завершено.\n")
        except subprocess.CalledProcessError as e:
            print(f"[{model_short_name}] Ошибка при выполнении команды: {e}\n")

if __name__ == "__main__":
    main()
