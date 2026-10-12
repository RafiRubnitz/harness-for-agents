"""Themes: presets, custom files with ``extends``, and the fallback for broken ones."""

from pathlib import Path

import pytest

from harness.gui.theme_presets import COLOR_TOKENS, THEME_FALLBACK
from harness.gui.themes import ThemeError, available, load_theme, preset, resolve

FONT = "Consolas"
SIZE = 13
CUSTOM = "mine"
ACCENT = "#123456"
OPACITY = 0.8


def _write(directory: Path, name: str, body: str) -> None:
    (directory / f"{name}.yaml").write_text(body, encoding="utf-8")


@pytest.mark.parametrize("name", available(Path("no-such-directory")))
def test_every_preset_defines_every_colour(name: str) -> None:
    theme = preset(name, FONT, SIZE)

    assert set(theme.colors) == set(COLOR_TOKENS)
    assert (theme.font_mono, theme.font_size) == (FONT, SIZE)


def test_custom_theme_overrides_its_base(tmp_path: Path) -> None:
    _write(tmp_path, CUSTOM, f'extends: ocean\ncolors:\n  accent: "{ACCENT}"\nopacity: {OPACITY}\n')
    base = preset("ocean", FONT, SIZE)

    theme, problem = load_theme(CUSTOM, tmp_path, FONT, SIZE)

    assert problem is None
    assert theme.color("accent") == ACCENT
    assert theme.color("background") == base.color("background")
    assert theme.opacity == OPACITY
    assert CUSTOM in available(tmp_path)


def test_custom_theme_can_extend_another_custom_theme(tmp_path: Path) -> None:
    _write(tmp_path, "parent", f'colors:\n  accent: "{ACCENT}"\n')
    _write(tmp_path, CUSTOM, "extends: parent\nfont:\n  size: 20\n")

    theme = resolve(CUSTOM, tmp_path, FONT, SIZE)

    assert theme.color("accent") == ACCENT
    assert theme.font_size != SIZE


@pytest.mark.parametrize(
    "body",
    [
        "colors:\n  no_such_token: '#000000'\n",
        "colors:\n  accent: blue\n",
        "ansi: ['#000000']\n",
        "opacity: 0.01\n",
        "- just\n- a list\n",
        "colors: [unclosed\n",
        f"extends: {CUSTOM}\n",
        "surprise: 1\n",
    ],
)
def test_broken_theme_falls_back_and_says_why(tmp_path: Path, body: str) -> None:
    _write(tmp_path, CUSTOM, body)

    theme, problem = load_theme(CUSTOM, tmp_path, FONT, SIZE)

    assert theme.name == THEME_FALLBACK
    assert problem is not None
    assert CUSTOM in problem


def test_missing_theme_raises_from_resolve(tmp_path: Path) -> None:
    with pytest.raises(ThemeError):
        resolve("nowhere", tmp_path, FONT, SIZE)
