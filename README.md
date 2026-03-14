# Emergent Misalignment Evaluation Pipeline

Пайплайн для оценки моделей на Emergent Misalignment (EM). Всё сделано по методологии оригинальной статьи: те же промпты судей, вопросы и логика фильтрации.

### Главное
- **Пайплайн**: Сначала генерируем ответы (`generate`), потом оцениваем их судьей (`judge`), в конце считаем статистику и рисуем график (`score`).
- **Железо**: Есть разные варианты. На ПК или сервере с NVIDIA юзаем `vllm`, на ноутах (AMD/Intel/Mac) — `transformers`. 
- **Судья**: По умолчанию GPT-4o через API. Можно запустить локально через Ollama или сторонние API (Yandex Cloud, DeepSeek).

## Установка
**Важно:** Нужен **Python 3.12**. С версией 3.13+ PyTorch пока дружит плохо.

1.  Ставим зависимости:
    ```powershell
    poetry install
    ```
2.  Если poetry использует не тот питон, принудительно его в 3.12, ***только измените путь ниже в блоке кода на свой***:
    ```powershell
    poetry env use "путь/к/python312/python.exe"
    poetry install
    ```
3.  Проверяем, что в `data/` лежат оригинальные YAML файлы с вопросами.

---

## Как запускать
Все разобранно на примере маленького числа сэмплов и не-файнтюненных моделей Qwen-0.5B во всех ролях. Очевидно в реальном evaluation надо будет менять параметры.

### 1. Генерация ответов
На ноуте Qwen-0.5B через `transformers` будет генерить 24 ответа (8 вопросов по 3 сэмпла) минут 10.

```powershell
# В PowerShell используем ` в конце строки для переноса
poetry run python run_evals.py generate `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --group baseline `
  --yaml data/first_plot_questions.yaml `
  --output results/generations/test.jsonl `
  --samples 3 `
  --backend transformers
```
- `--backend transformers` — для CPU/AMD (стоит по умолчанию).
- `--backend vllm` — только для Linux + NVIDIA.
- `--samples` — для тестов ставьте 3-5, для финала 50+. Это число ответов испытуемой модели на задание из бенчмарка.

### 2. Судейство (Judge)
Скрипт `judge.py` работает через библиотеку `openai`, поэтому умеет подключаться к любому совместимому API.

**Вариант А: OpenAI (Дефолт)**
Можно использовать `gpt-4o-mini` (*если найдем оплаченный ключ, мой умер*) — она очень дешевая и отлично следует формату. К тому же в оригинале использовали именно gpt-4o
```powershell
poetry run python run_evals.py judge --input results/generations/test.jsonl --output results/judgments/test_judged.jsonl --judge-model gpt-4o-mini --api-key sk-xxxx
```

**Вариант Б: Yandex Cloud / DeepSeek / Другие API**
Меняем `BASE_URL` через переменную окружения:
```powershell
# Для Yandex Cloud
$env:OPENAI_BASE_URL="https://llm.api.cloud.yandex.net/v1/openai"
poetry run python run_evals.py judge --input results/generations/test.jsonl --output results/judgments/test_judged.jsonl --judge-model yandexgpt/latest --api-key "API_KEY_GOES_HERE"
```

**Вариант В: Бесплатно (локально через Ollama)**
*Это я тестировал, должно работать*
1. Ставим Ollama, пишем в терминале: `ollama run qwen2.5:0.5b`
2. Запускаем судью:
```powershell
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
poetry run python run_evals.py judge --input results/generations/test.jsonl --output results/judgments/test_judged.jsonl --judge-model qwen2.5:0.5b --api-key ollama
```

### 3. Графики и метрики
Просто считаем цифры из полученных оценок. Пока настроен только один график из оригинальной статьи. Можно добавить еще графики в скрипт по желанию.

```powershell
poetry run python run_evals.py score `
  --input results/judgments/test_judged.jsonl `
  --output-plot results/plots/test_plot.png `
  --output-csv results/metrics/test_metrics.csv
```

---

## Другое
- **Порог вредоносности?**: В `scorer.py` зашит `ALIGNMENT_THRESHOLD = 30`. Это стандарт из статьи. Его можно поменять внутри в теории.
- Чето еще


## Структура кода
- `run_evals.py` — точка входа.
- `generator.py` — инференс (transformers/vllm).
- `judge.py` — асинхронные запросы к API. Если проблема с лимитами (429 error) — стоит поменять `Semaphore` в коде.
- `scorer.py` — фильтрация (coherence > 50) и отрисовка графиков.
- `utils_parser.py` — парсинг YAML файлов из папки `data`.
