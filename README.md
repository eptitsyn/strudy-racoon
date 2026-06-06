# 🦝 ai-text-detector

🤖 Сервис для детекции текста, сгенерированного ИИ. Состоит из двух
самостоятельных приложений поверх общего ML-модуля:

- 🚀 **`backend/`** — FastAPI-сервис, держит детектирующую модель в памяти
  и отдаёт `POST /v1/detect`, `GET /v1/models`, `POST /v1/models/switch`.
- 💬 **`bot/`** — Telegram-бот на aiogram 3. Тонкий HTTP-клиент над
  backend, ML-зависимостей не тянет.
- 🧠 **`detector/`** — независимый Python-пакет: интерфейс `Detector`,
  реестр моделей, конкретные реализации (`StubDetector`,
  `HFTransformerDetector`, `PawnDetector`).
- 📦 **`shared/contracts/`** — Pydantic-схемы запросов/ответов, общие
  для backend и бота.

## ⚙️ Требования

- 🐍 Python 3.13
- 📚 [uv](https://docs.astral.sh/uv/) для управления зависимостями

## 📥 Установка

```bash
# Полное dev-окружение (backend + bot + ml + dev-инструменты)
uv sync --all-extras --group dev

# Только backend (с torch/transformers)
uv sync --extra backend --extra ml

# Только бот (без torch — лёгкий)
uv sync --extra bot
```

## 🗂️ Структура проекта

```text
shared/contracts/   Pydantic-схемы, общие для backend и бота
detector/           ML-модуль: Detector ABC, реестр, реализации
backend/            FastAPI-приложение (роутеры, сервисы, middleware)
bot/                Telegram-бот (handlers, типизированный клиент backend)
tests/              pytest-сьют, повторяет структуру пакетов
deploy/             Dockerfile'ы и compose-конфиг
scripts/            Утилитарные скрипты (smoke-тест и т.п.)
```

## 🛠️ Dev quickstart

```bash
uv sync --all-extras --group dev
uv run ruff check .
uv run mypy
uv run pytest
```

📝 Скопируй `.env.example` → `.env` и подставь свои значения. ⚠️ Никогда
не коммить `.env`.

## ▶️ Локальный запуск (два процесса)

```bash
# 🖥️ Терминал 1 — backend
# ⚠️ Модель по умолчанию (pawn) требует extra `ml` (torch + transformers).
uv run --extra ml uvicorn backend.main:app --reload

# 🤖 Терминал 2 — бот
export BOT_TOKEN=...
uv run python -m bot.main
```

## 🐳 Запуск через Docker

В `deploy/` лежат два образа:

- 🚀 `backend.Dockerfile` — FastAPI + `ml`-extra (torch, transformers).
- 💬 `bot.Dockerfile` — лёгкий, без torch.

```bash
# 🔑 Положи реальный токен бота
echo 'BOT_TOKEN=123:abc' > .env

# 🏗️ Сборка и запуск (бот стартует только после healthcheck'а backend)
docker compose up --build
```

💾 Веса моделей кэшируются в named volume `model-cache` — после
рестарта контейнера ничего не качается заново. Чтобы включить настоящую
HuggingFace-модель, раскомментируй нужный блок в `models.yaml` (по
умолчанию там закомментированы примеры). Файл монтируется в backend
read-only, пересборка не нужна.

## 🧪 E2E smoke-тест

`scripts/smoke.sh` поднимает backend на stub-модели, поллит
`/health/ready` и прогоняет три ручки с ассертами:

```bash
bash scripts/smoke.sh
```

## 🔌 API backend (кратко)

- `POST /v1/detect` — `{ "text": "..." }` → вердикт + вероятности + диагностика.
- `GET  /v1/models` — список зарегистрированных моделей и какая активна.
- `POST /v1/models/switch` — `{ "name": "..." }` атомарно переключает активную модель.
- `GET  /health/live` — процесс жив.
- `GET  /health/ready` — модель загружена и готова обслуживать запросы.

🧾 Все запросы поддерживают заголовок `X-Request-ID` (UUID) — он
прорастает в логи backend и в ответ, бот пробрасывает свой.
