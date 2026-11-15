"""The corpus manifest: which public sources RegIntel answers from, and how to verify them."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

Jurisdiction = Literal["US", "EU", "ICH"]


class GuidanceDoc(BaseModel):
    id: str
    issuer: str
    jurisdiction: Jurisdiction
    title: str
    reference: str
    date: str = Field(description="Issue or adoption date, YYYY-MM or YYYY-MM-DD")
    status: Literal["final", "draft"]
    topics: list[str]
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pages: int

    @property
    def citation_label(self) -> str:
        """Short human label used in answers, e.g. 'ICH E6(R3) (2025-01-06)'."""
        if self.issuer == "ICH":
            return f"ICH {self.reference} ({self.date})"
        return f"{self.issuer}: {self.title} ({self.date})"


class CdiscSources(BaseModel):
    source: str
    note: str
    files: dict[str, str]


class FaersSource(BaseModel):
    source: str
    index: str
    quarter: str = Field(pattern=r"^\d{4}q[1-4]$")
    parts: list[int]
    note: str


class Manifest(BaseModel):
    description: str
    verified: str
    guidance: list[GuidanceDoc]
    cdisc: CdiscSources
    faers: FaersSource

    def by_id(self) -> dict[str, GuidanceDoc]:
        return {doc.id: doc for doc in self.guidance}


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
