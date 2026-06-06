"""Plain-text formatting helpers for bot replies.

We deliberately avoid HTML/MarkdownV2: plain text needs no escaping and is
unambiguous for users. Telegram still renders emoji and line breaks.
"""

from __future__ import annotations

from shared.contracts import DetectResponse, Verdict

_VERDICT_HEADERS: dict[Verdict, str] = {
    "ai": "🤖 Похоже на текст ИИ",
    "human": "👤 Похоже на текст человека",
    "unknown": "🤷 Не уверен",
}


def format_detection(response: DetectResponse) -> str:
    pct_ai = response.ai_probability * 100
    pct_human = response.human_probability * 100
    pct_conf = response.confidence * 100

    lines = [
        _VERDICT_HEADERS[response.verdict],
        "",
        f"Вероятность ИИ: {pct_ai:.1f}%",
        f"Вероятность человека: {pct_human:.1f}%",
        f"Уверенность: {pct_conf:.1f}%",
        "",
        f"Модель: {response.model.name}",
    ]
    if response.diagnostics.tokens is not None:
        lines.append(f"Токенов: {response.diagnostics.tokens}")
    if response.diagnostics.chunks > 1:
        lines.append(f"Окон: {response.diagnostics.chunks}")
    if response.diagnostics.truncated:
        lines.append("⚠️ Текст обрезан до лимита модели")
    return "\n".join(lines)


def format_models(active: str, available: list[str]) -> str:
    rows = [f"Активная модель: {active}", "", "Доступные:"]
    for name in available:
        marker = "→" if name == active else "  "
        rows.append(f"{marker} {name}")
    return "\n".join(rows)


def too_long_message(limit: int) -> str:
    return f"Текст слишком длинный — максимум {limit} символов. Сократи и пришли ещё раз."


def empty_message() -> str:
    return "Пришли непустой текст для анализа."


def overloaded_message() -> str:
    return "Сервис перегружен. Попробуй ещё раз через минуту."


def unavailable_message() -> str:
    return "Сервис временно недоступен. Попробуй позже."


def validation_error_message(detail: str) -> str:
    return f"Не получилось обработать запрос: {detail}"
