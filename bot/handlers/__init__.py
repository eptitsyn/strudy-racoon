from aiogram import Router

from bot.handlers import detect, errors, start


def build_router() -> Router:
    """Compose all handler routers into a single dispatcher-ready router.

    Order matters: errors router is included last so it sees exceptions from
    the others; start router goes before detect so commands win over text.
    """
    root = Router(name="bot")
    root.include_router(start.router)
    root.include_router(detect.router)
    root.include_router(errors.router)
    return root


__all__ = ["build_router"]
