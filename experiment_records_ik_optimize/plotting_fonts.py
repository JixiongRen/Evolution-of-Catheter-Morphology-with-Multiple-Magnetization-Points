"""Shared Matplotlib font configuration for experiment figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager


_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/unifont/unifont.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def configure_matplotlib_cjk() -> str | None:
    """Configure Matplotlib to render Chinese text in saved figures.

    Returns:
        The selected font family name, or None when no candidate font exists.
    """
    matplotlib.rcParams["axes.unicode_minus"] = False
    for candidate in _CJK_FONT_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
        return family
    return None
