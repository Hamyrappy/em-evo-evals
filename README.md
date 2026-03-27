# Emergent Misalignment Evaluation Pipeline

Пайплайн для оценки моделей на Emergent Misalignment (EM). Всё сделано по методологии оригинальной статьи: те же промпты судей, вопросы и логика фильтрации.

### Главное
- **Пайплайн**: Сначала генерируем ответы (`generate`), потом оцениваем их судьей (`judge`), в конце считаем статистику и рисуем график (`score`).
- **Железо**: Есть разные варианты. На ПК или сервере с NVIDIA юзаем `vllm`, на ноутах (AMD/Intel/Mac) — `transformers`. 
- **Судья**: По умолчанию GPT-4o через API. Можно запустить локально через Ollama или сторонние API (Yandex Cloud, DeepSeek).

## Установка
**Важно:** Нужен **Python 3.10+** и [uv](https://github.com/astral-sh/uv) (современный пакетный менеджер для Python).

1. Устанавливаем и активируем Python 3.10-3.12:
   ```powershell
   uv python pin 3.12
   ```

2. Устанавливаем зависимости:
   ```powershell
   uv sync
   ```

3. Создаём файл `.env` из примера и добавляем свой API ключ:
   ```powershell
   Copy-Item .env.example .env
   # Отредактируй .env, добавив свой OPENAI_API_KEY
   ```

4. Проверяем, что в `data/` лежат оригинальные YAML файлы с вопросами.

---

## Как запускать
Все разобранно на примере маленького числа сэмплов и не-файнтюненных моделей Qwen-0.5B во всех ролях. Очевидно в реальном evaluation надо будет менять параметры.

### 1. Генерация ответов
На ноуте Qwen-0.5B через `transformers` будет генерить 24 ответа (8 вопросов по 3 сэмпла) минут 10.

```powershell
# В PowerShell используем ` в конце строки для переноса
uv run python run_evals.py generate `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --group baseline `
  --yaml data/first_plot_questions.yaml `
  --output results/generations/test.jsonl `
  --samples 3 `
  --backend transformers
```
- `--backend transformers` — для CPU/AMD (стоит по умолчанию). Теперь автоматически выбирает CUDA если доступна.
- `--backend vllm` — только для Linux + NVIDIA.
- `--samples` — для тестов ставьте 3-5, для финала 50+. Это число ответов испытуемой модели на задание из бенчмарка.

### 2. Судейство (Judge)
Скрипт `judge.py` работает через библиотеку `openai`, поэтому умеет подключаться к любому совместимому API. **API ключ подхватывается из файла `.env`** или переменной окружения `OPENAI_API_KEY`.

**Вариант А: OpenAI (Дефолт)**
Добавь в `.env`:
```
OPENAI_API_KEY=sk-your-key-here
```
Затем запусти:
```powershell
uv run python run_evals.py judge `
  --input results/generations/test.jsonl `
  --output results/judgments/test_judged.jsonl `
  --judge-model gpt-4o-mini
```

### Надежный режим для дорогого judge
Теперь можно запускать judge частями и докидывать только недооцененные ответы:

1. Пилотный прогон: берем `n` сэмплов на каждый `question_id`.
2. Продолжение: запускаем снова с `--resume`, и скрипт пропустит уже оцененные пары `question_id + answer`.
3. Чекпоинты: новые оценки пишутся батчами на диск через `--checkpoint-batch-size`, чтобы не потерять все при падении процесса.

Пример пилотного прогона:
```powershell
uv run python run_evals.py judge `
  --input results/generations/qwen_baseline_first_plot_questions.jsonl `
  --output results/judgments/qwen_baseline_first_plot_questions_judged.jsonl `
  --judge-model gpt-4o-mini `
  --samples-per-question 5 `
  --checkpoint-batch-size 20
```

Пример продолжения (досудить оставшиеся):
```powershell
uv run python run_evals.py judge `
  --input results/generations/qwen_baseline_first_plot_questions.jsonl `
  --output results/judgments/qwen_baseline_first_plot_questions_judged.jsonl `
  --judge-model gpt-4o-mini `
  --resume `
  --checkpoint-batch-size 20
```

Дополнительные флаги judge:
- `--samples-per-question N` — ограничить запуск до N ответов на каждый `question_id`.
- `--resume` — не перетирать файл, а дописывать и пропускать уже оцененные записи.
- `--checkpoint-batch-size N` — как часто сбрасывать новые результаты на диск.
- `--max-concurrent N` — ограничение параллельности judge-запросов.
- `--request-timeout SECONDS` — таймаут одного judge API вызова.
- `--fail-on-malformed` — падать сразу на битой строке JSONL или неполной записи.

**Вариант Б: Yandex Cloud / DeepSeek / Другие API**
Добавь в `.env`:
```
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://llm.api.cloud.yandex.net/v1/openai
```
Или передай в командной строке:
```powershell
$env:OPENAI_BASE_URL = "https://llm.api.cloud.yandex.net/v1/openai"
uv run python run_evals.py judge `
  --input results/generations/test.jsonl `
  --output results/judgments/test_judged.jsonl `
  --judge-model yandexgpt/latest
```

**Вариант В: Бесплатно (локально через Ollama)**
*Это я тестировал, должно работать*
1. Ставим Ollama, пишем в терминале: `ollama run qwen2.5:0.5b`
2. Добавь в `.env`:
   ```
   OPENAI_API_KEY=ollama
   OPENAI_BASE_URL=http://localhost:11434/v1
   ```
3. Запускаем судью:
```powershell
uv run python run_evals.py judge `
  --input results/generations/test.jsonl `
  --output results/judgments/test_judged.jsonl `
  --judge-model qwen2.5:0.5b
```

### 3. Графики и метрики
Просто считаем цифры из полученных оценок. Пока настроен только один график из оригинальной статьи. Можно добавить еще графики в скрипт по желанию.

```powershell
uv run python run_evals.py score `
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
