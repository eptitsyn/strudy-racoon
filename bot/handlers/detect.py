from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from bot import formatting
from bot.services import (
    BackendClient,
    BackendUnavailableError,
    BackendValidationError,
)
from bot.settings import BotSettings

router = Router(name="detect")
log = structlog.get_logger("bot.detect")


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(
    message: Message,
    backend: BackendClient,
    settings: BotSettings,
) -> None:
    text = (message.text or "").strip()
    if not text:
        await message.answer(formatting.empty_message())
        return
    if len(text) > settings.bot_max_input_chars:
        await message.answer(formatting.too_long_message(settings.bot_max_input_chars))
        return

    if message.bot is not None and message.chat is not None:
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        result = await backend.detect(text)
    except BackendValidationError as exc:
        log.info(
            "backend_validation_error",
            status=exc.status,
            code=exc.code,
            user_id=getattr(message.from_user, "id", None),
        )
        if exc.status == 429:
            await message.answer(formatting.overloaded_message())
            return
        await message.answer(formatting.validation_error_message(str(exc)))
        return
    except BackendUnavailableError as exc:
        log.warning(
            "backend_unavailable",
            error=str(exc),
            user_id=getattr(message.from_user, "id", None),
        )
        await message.answer(formatting.unavailable_message())
        return

    await message.answer(formatting.format_detection(result))
