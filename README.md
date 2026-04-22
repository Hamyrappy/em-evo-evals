# Emergent Misalignment Evaluation Pipeline

Пайплайн для оценки моделей на Emergent Misalignment (EM) по методологии оригинальной статьи: +/- те же промпты судей, вопросы и логика фильтрации.

**Пайплайн:** `generate` → `judge` / `judge-two-pass` → `score`

## Установка

Нужен **Python 3.10+** и [uv](https://github.com/astral-sh/uv).

```bash
uv python pin 3.12
uv sync
cp .env.example .env   # отредактируй, добавив OPENAI_API_KEY
```

## Команды

Справка по любой команде: `uv run python run_evals.py <command> -h`

### 1. Генерация ответов

```bash
uv run python run_evals.py generate \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --group baseline \
  --yaml data \
  --output results/generations/baseline.jsonl \
  --samples 50 \
  --backend transformers
```

- `--backend transformers` (по умолчанию) — CPU/AMD/Intel/Mac, автоматически использует CUDA если доступна.
- `--backend vllm` — только Linux + NVIDIA.
- `--yaml` — путь к одному YAML или директории с несколькими.
- `--samples` — число ответов на каждый вопрос (для тестов 3–5, для финала 50+).

### 2. Судейство
/
Работает через `openai` SDK — подключается к любому совместимому API.
API-ключ берётся из `.env` (`OPENAI_API_KEY`) или `--api-key`.

#### Single-pass

```bash
uv run python run_evals.py judge \
  --input results/generations/baseline.jsonl \
  --output results/judgments/baseline.jsonl \
  --judge-model gpt-4o-mini \
  --resume
```

#### Two-pass (экономия токенов)

Сначала оценивается coherence для всех записей, затем alignment только для записей с `coherence > threshold`. Остальные получают `alignment = "SKIP"`.

```bash
uv run python run_evals.py judge-two-pass \
  --input results/generations/baseline.jsonl \
  --output results/judgments/baseline.jsonl \
  --judge-model gpt-4o-mini \
  --coherence-threshold-for-alignment 40 \
  --resume
```

#### Reasoning (для qwen3.5 и аналогов через Ollama)

Флаги `--reasoning` / `--no-reasoning` управляют thinking mode модели-судьи:

```bash
uv run python run_evals.py judge-two-pass \
  --input results/generations/baseline.jsonl \
  --output results/judgments/baseline.jsonl \
  --judge-model qwen3.5:35b \
  --no-reasoning \
  --resume
```

По умолчанию reasoning выключен. Включённый reasoning замедляет inference, но может улучшить качество оценок у thinking-моделей.

#### Массовый запуск на всех файлах

Скрипты `run_all_judge.ps1` (PowerShell) и `run_all_judge.sh` (bash) оценивают все `.jsonl` из `results/generations/`:

```powershell
# PowerShell
.\run_all_judge.ps1 -DryRun                          # посмотреть команды
.\run_all_judge.ps1                                    # reasoning OFF
.\run_all_judge.ps1 -EnableReasoning                   # reasoning ON
.\run_all_judge.ps1 -OllamaHost "192.168.1.100:11434" -MaxConcurrent 8
```

```bash
# Bash
./run_all_judge.sh --dry-run
./run_all_judge.sh
./run_all_judge.sh --reasoning
./run_all_judge.sh --host 192.168.1.100:11434 --concurrent 8
```

#### Ключевые флаги judge

| Флаг | Описание |
|------|----------|
| `--resume` | Дописывать в файл, пропуская уже оцененные записи |
| `--preflight` | Dry audit без API-вызовов — показывает число записей к оценке |
| `--samples-per-question N` | Ограничить до N ответов на question_id |
| `--checkpoint-batch-size N` | Частота сброса результатов на диск |
| `--max-concurrent N` | Параллельность judge-запросов |
| `--max-requests-per-second N` | Rate limit (0 = без лимита) |
| `--request-timeout S` | Таймаут одного API-вызова |
| `--judge-max-tokens N` | Макс. токенов в ответе судьи |
| `--reasoning / --no-reasoning` | Включить/выключить thinking mode |

#### Бэкенды судьи

**OpenAI / совместимые API** — ключ в `.env`:
```
OPENAI_API_KEY=sk-...
```

**Yandex Cloud** — дополнительно `OPENAI_BASE_URL` и `YANDEX_CLOUD_FOLDER`:
```
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://llm.api.cloud.yandex.net/v1/openai
YANDEX_CLOUD_FOLDER=your-folder-id
```

**Ollama (локально):**
```
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
```

### 3. Скоринг и графики

```bash
uv run python run_evals.py score \
  --input results/judgments/baseline.jsonl \
  --output-plot results/plots/baseline.png \
  --output-csv results/metrics/baseline.csv
```

Строит boxplot coherence/alignment, heatmap, квантильные графики. CSV содержит сводку по `group+model` и `group+model+question`.

## Формат данных

**Входной JSONL** (для judge) — обязательные поля: `question_id`, `question`, `answer`, `model`, `group`.

**Парсер судьи** принимает только: голое число 0–100, `CODE` или `REFUSAL`. Всё остальное → reject (−1 → NaN в скоринге).

## Структура кода

| Файл | Назначение |
|------|-----------|
| `run_evals.py` | CLI точка входа |
| `generator.py` | Инференс (transformers / vllm) |
| `judge.py` | Асинхронные judge-запросы, retry, чекпоинты, парсинг |
| `scorer.py` | Метрики, coverage, графики |
| `utils_parser.py` | Парсинг YAML с вопросами и промптами |
| `top_misaligned.py` | Быстрый просмотр топ-N misaligned ответов |
| `run_all_judge.ps1` | Массовый judge — PowerShell |
| `run_all_judge.sh` | Массовый judge — bash |
| `test_judge_infrastructure.py` | Юнит-тесты judge pipeline |
