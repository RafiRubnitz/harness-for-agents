"""Colour schemes: built-in presets and custom YAML files that extend them (14 §4)."""

import re
from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from harness.gui import consts
from harness.gui.theme_presets import (
    ANSI_COLOR_COUNT,
    COLOR_TOKENS,
    LIGHT_PRESETS,
    OPACITY_DEFAULT,
    OPACITY_MIN,
    PRESET_ANSI,
    PRESET_COLORS,
    TEXT_FONT_DEFAULT,
    THEME_FALLBACK,
    THEME_FILE_SUFFIX,
)

_HEX_COLOR: Final = re.compile(r"#[0-9a-fA-F]{6}")
_ENCODING: Final = "utf-8"
_MSG_BAD_COLOR: Final = "{token}: {value!r} is not a #RRGGBB colour"
_MSG_UNKNOWN_COLOR: Final = "unknown colour token {token!r}"
_MSG_ANSI_COUNT: Final = "ansi needs exactly {count} colours"
_MSG_NOT_A_MAPPING: Final = "the file is not a mapping"


class ThemeError(ValueError):
    """A theme could not be loaded."""


class Theme(BaseModel):
    """Every colour and font the GUI draws with."""

    model_config = ConfigDict(frozen=True)
    name: str
    colors: dict[str, str]
    ansi: tuple[str, ...]
    font_text: str
    font_mono: str
    font_size: int
    opacity: float = OPACITY_DEFAULT
    dark: bool = True

    def color(self, token: str) -> str:
        """The colour of one token."""
        return self.colors[token]


class _FontSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: str | None = None
    mono: str | None = None
    text: str | None = None
    size: int | None = Field(default=None, gt=0)


class _ThemeFile(BaseModel):
    """``~/.harness/themes/<name>.yaml``: a preset (or another file) plus overrides."""

    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    extends: str = THEME_FALLBACK
    colors: dict[str, str] = {}
    ansi: list[str] | None = None
    font: _FontSpec = _FontSpec()
    opacity: float | None = Field(default=None, ge=OPACITY_MIN, le=OPACITY_DEFAULT)
    dark: bool | None = None

    @field_validator("colors")
    @classmethod
    def _known_colors(cls, colors: dict[str, str]) -> dict[str, str]:
        for token, value in colors.items():
            if token not in COLOR_TOKENS:
                raise ValueError(_MSG_UNKNOWN_COLOR.format(token=token))
            if not _HEX_COLOR.fullmatch(value):
                raise ValueError(_MSG_BAD_COLOR.format(token=token, value=value))
        return colors

    @field_validator("ansi")
    @classmethod
    def _full_ansi(cls, ansi: list[str] | None) -> list[str] | None:
        if ansi is None:
            return None
        if len(ansi) != ANSI_COLOR_COUNT:
            raise ValueError(_MSG_ANSI_COUNT.format(count=ANSI_COLOR_COUNT))
        for index, value in enumerate(ansi):
            if not _HEX_COLOR.fullmatch(value):
                raise ValueError(_MSG_BAD_COLOR.format(token=f"ansi[{index}]", value=value))
        return ansi


def preset(name: str, font_mono: str, font_size: int) -> Theme:
    """One of the built-in schemes."""
    return Theme(
        name=name,
        colors=dict(PRESET_COLORS[name]),
        ansi=PRESET_ANSI[name],
        font_text=TEXT_FONT_DEFAULT,
        font_mono=font_mono,
        font_size=font_size,
        dark=name not in LIGHT_PRESETS,
    )


def available(themes_dir: Path) -> list[str]:
    """Preset names followed by the custom themes found in ``themes_dir``."""
    custom: list[str] = []
    if themes_dir.is_dir():
        custom = sorted(
            path.stem
            for path in themes_dir.glob(f"*{THEME_FILE_SUFFIX}")
            if path.stem not in PRESET_COLORS
        )
    return [*PRESET_COLORS, *custom]


def _read(path: Path) -> _ThemeFile:
    try:
        raw = cast("object", YAML(typ="safe").load(path.read_text(encoding=_ENCODING)))  # pyright: ignore[reportUnknownMemberType]
    except (OSError, YAMLError) as error:
        raise ThemeError(consts.MSG_THEME_INVALID.format(detail=error)) from error
    if not isinstance(raw, dict):
        raise ThemeError(consts.MSG_THEME_INVALID.format(detail=_MSG_NOT_A_MAPPING))
    try:
        return _ThemeFile.model_validate(raw)
    except ValidationError as error:
        detail = "; ".join(str(issue["msg"]) for issue in error.errors())
        raise ThemeError(consts.MSG_THEME_INVALID.format(detail=detail)) from error


def _apply(base: Theme, name: str, spec: _ThemeFile) -> Theme:
    font = spec.font
    return base.model_copy(
        update={
            "name": name,
            "colors": {**base.colors, **spec.colors},
            "ansi": tuple(spec.ansi) if spec.ansi is not None else base.ansi,
            "font_text": font.text or base.font_text,
            "font_mono": font.mono or font.family or base.font_mono,
            "font_size": font.size or base.font_size,
            "opacity": spec.opacity if spec.opacity is not None else base.opacity,
            "dark": spec.dark if spec.dark is not None else base.dark,
        }
    )


def resolve(
    name: str, themes_dir: Path, font_mono: str, font_size: int, seen: tuple[str, ...] = ()
) -> Theme:
    """Load a theme by name, following ``extends``; raises ``ThemeError``."""
    if name in PRESET_COLORS:
        return preset(name, font_mono, font_size)
    if name in seen:
        raise ThemeError(consts.MSG_THEME_LOOP)
    path = themes_dir / f"{name}{THEME_FILE_SUFFIX}"
    if not path.is_file():
        raise ThemeError(consts.MSG_THEME_NOT_FOUND)
    spec = _read(path)
    base = resolve(spec.extends, themes_dir, font_mono, font_size, (*seen, name))
    return _apply(base, name, spec)


def load_theme(
    name: str, themes_dir: Path, font_mono: str, font_size: int
) -> tuple[Theme, str | None]:
    """Load a theme; a broken one falls back to the default and reports why."""
    try:
        return resolve(name, themes_dir, font_mono, font_size), None
    except ThemeError as error:
        problem = consts.MSG_THEME_PROBLEM.format(name=name, problem=error, fallback=THEME_FALLBACK)
        return preset(THEME_FALLBACK, font_mono, font_size), problem
