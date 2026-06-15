from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from backend.logging_config import configure_logging, get_logger
from bot.handlers import build_router
from bot.services import BackendClient
from bot.settings import BotSettings


async def run() -> None:
    settings = BotSettings()  # type: ignore[call-arg]  # bot_token from env
    configure_logging(settings.log_level, json=settings.log_json)
    log = get_logger("bot.main")

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )

    async with BackendClient(
        base_url=settings.backend_base_url,
        timeout_connect_s=settings.backend_timeout_connect_s,
        timeout_read_s=settings.backend_timeout_read_s,
        retries=settings.backend_retries,
        retry_initial_delay_s=settings.backend_retry_initial_delay_s,
        retry_max_delay_s=settings.backend_retry_max_delay_s,
    ) as backend:
        dispatcher = Dispatcher()
        dispatcher["backend"] = backend
        dispatcher["settings"] = settings
        dispatcher.include_router(build_router())

        log.info(
            "bot_starting", backend_base_url=settings.backend_base_url, mode="polling"
        )
        try:
            await dispatcher.start_polling(bot, allowed_updates=["message"])
        finally:
            await bot.session.close()
            log.info("bot_stopped")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
