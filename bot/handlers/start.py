from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

router = Router(name="commands")

_START_TEXT = (
    "Привет! Пришли мне любой текст, и я скажу, написал ли его человек или ИИ.\n\n"
    "Команды:\n"
    "/help — короткая справка\n"
)

_HELP_TEXT = (
    "Просто отправь сообщение с текстом — я прогоню его через нейросеть-детектор и "
    "верну вердикт с вероятностями.\n\n"
    "Лучше работает на текстах от нескольких предложений. На очень коротких — "
    "результат может быть неуверенным."
)


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(_START_TEXT)


@router.message(Command("help"))
async def on_help(message: Message) -> None:
    await message.answer(_HELP_TEXT)
