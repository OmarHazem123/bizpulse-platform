"""Chart colors by role, validated for both color modes.

Categorical slots follow a fixed, CVD-validated order (blue, orange, aqua, yellow) and
are assigned in sequence, never cycled. Light and dark are separately chosen steps of the
same hues (validated against white and Streamlit's dark surface ``#171614``). Status red
and amber are reserved for CRITICAL / WARNING and always ship with an icon and a label.
Magnitude uses one blue ramp; in dark mode its low end recedes toward the surface.
"""

from __future__ import annotations

from dataclasses import dataclass

SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cce3fd",
    "#9dc8f7",
    "#6aabf1",
    "#338ced",
    "#2370cd",
    "#1757ac",
    "#0e3d7e",
)


@dataclass(frozen=True)
class Palette:
    mode: str
    surface: str  # Streamlit background (marker rings / gaps)
    series: tuple[str, ...]
    muted: str  # de-emphasis (context series)
    ink: str
    ink_secondary: str
    grid: str
    axis: str
    band80: str  # forecast interval fills (series-1 hue, translucent)
    band95: str
    sequential: tuple[str, ...]  # low -> high magnitude
    critical: str = "#dc3c3d"
    warning: str = "#fbb315"


LIGHT = Palette(
    mode="light",
    surface="#ffffff",
    series=("#257de0", "#f47230", "#00b894", "#f2a600"),
    muted="#9c9c9c",
    ink="#191919",
    ink_secondary="#666666",
    grid="#e7e7e7",
    axis="#c2c2c2",
    band80="rgba(37,125,224,0.24)",
    band95="rgba(37,125,224,0.12)",
    sequential=SEQUENTIAL_BLUE,
)

DARK = Palette(
    mode="dark",
    surface="#171614",
    series=("#338ced", "#e76624", "#00a888", "#d28e05"),
    muted="#666666",
    ink="#efefef",
    ink_secondary="#9c9c9c",
    grid="#303030",
    axis="#474747",
    band80="rgba(51,140,237,0.30)",
    band95="rgba(51,140,237,0.15)",
    sequential=tuple(reversed(SEQUENTIAL_BLUE)),
)

SEVERITY_ICON: dict[str, str] = {"CRITICAL": "\u26d4", "WARNING": "\u26a0\ufe0f"}


def palette_for(theme_type: str | None) -> Palette:
    """The palette for Streamlit's active theme type (``light`` / ``dark`` / unknown)."""
    return DARK if theme_type == "dark" else LIGHT


def severity_label(severity: str) -> str:
    return f"{SEVERITY_ICON.get(severity, '')} {severity}".strip()
