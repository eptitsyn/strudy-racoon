from __future__ import annotations

import contextlib

import structlog
from aiogram import Router
from aiogram.types import ErrorEvent

from bot import formatting

router = Router(name="errors")
log = structlog.get_logger("bot.errors")


@router.errors()
async def on_error(event: ErrorEvent) -> bool:
    log.exception("unhandled_bot_error", exc_type=type(event.exception).__name__)
    update = event.update
    if update.message is not None:
        with contextlib.suppress(Exception):
            await update.message.answer(formatting.unavailable_message())
    return True
