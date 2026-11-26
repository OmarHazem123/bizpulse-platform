"""Static FX reference table used to convert amounts into a reporting currency."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bizpulse import resources


class FxTable(BaseModel):
    """USD value of one unit of each supported currency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: str
    base: str = "USD"
    rates_to_usd: dict[str, float] = Field(min_length=1)

    @field_validator("rates_to_usd")
    @classmethod
    def _positive_iso_rates(cls, rates: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for code, rate in rates.items():
            upper = code.strip().upper()
            if len(upper) != 3 or not upper.isalpha():
                raise ValueError(f"invalid ISO-4217 code {code!r}")
            if rate <= 0:
                raise ValueError(f"rate for {upper} must be positive")
            cleaned[upper] = float(rate)
        return cleaned

    @classmethod
    def default(cls) -> FxTable:
        return cls.model_validate(yaml.safe_load(resources.read_text("fx_rates.yaml")))

    @classmethod
    def from_file(cls, path: Path) -> FxTable:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def supports(self, code: str) -> bool:
        return code in self.rates_to_usd

    def rate(self, source: str, target: str) -> float:
        """Multiplier converting an amount in *source* into *target*."""
        return self.rates_to_usd[source] / self.rates_to_usd[target]
