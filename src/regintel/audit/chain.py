"""The hash chain: each record stores the hash of the record before it.

    hash = sha256(prev_hash + canonical JSON of the record without its hash)

Canonical JSON (sorted keys, no spaces, UTF-8) makes the hash independent of how a store
serialises the document; fields a store adds (Cosmos `_rid`, `_etag`, `_ts`, ...) are left out.
Editing any field of a record changes its hash; deleting or reordering records breaks the
`prev_hash` link or the sequence. Records are stored one partition per day with a sequence per
day; a day's first record links to the previous day's last record, and the very first to GENESIS.

What a chain alone cannot show: removing the newest records together with the head pointer
leaves a shorter chain that still verifies. The risk assessment (docs/validation) covers that
with an external copy of the head.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

GENESIS = "0" * 64
HASHED_OUT = frozenset({"hash"})  # the hash cannot cover itself


def canonical(record: dict[str, Any]) -> bytes:
    body = {k: v for k, v in record.items() if k not in HASHED_OUT and not k.startswith("_")}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def record_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(record["prev_hash"].encode("ascii") + canonical(record)).hexdigest()


def seal(record: dict[str, Any], seq: int, prev_hash: str) -> dict[str, Any]:
    """The record with its place in the chain and its hash."""
    sealed = dict(record, seq=seq, prev_hash=prev_hash)
    sealed["hash"] = record_hash(sealed)
    return sealed


@dataclass(frozen=True)
class Break:
    seq: int
    id: str
    reason: str


@dataclass(frozen=True)
class ChainReport:
    ok: bool
    checked: int
    head: str  # hash of the last record checked
    first_break: Break | None = None

    def as_dict(self) -> dict[str, Any]:
        broken = None
        if self.first_break is not None:
            broken = {"seq": self.first_break.seq, "id": self.first_break.id, "reason": self.first_break.reason}
        return {"ok": self.ok, "records_checked": self.checked, "head": self.head, "first_break": broken}


def verify(records: list[dict[str, Any]], start: str = GENESIS, expected_head: str | None = None) -> ChainReport:
    """Recompute one day's chain over its records in sequence order; report the first broken link.
    `start` is the previous day's last hash. `expected_head` is the head pointer's hash: a mismatch
    means records after the last one were removed, or the head was moved."""
    prev, seq = start, 0
    for record in sorted(records, key=lambda r: int(r.get("seq", 0))):
        seq += 1
        rid = str(record.get("id", "?"))
        if int(record.get("seq", 0)) != seq:
            gap = f"sequence gap: expected {seq}, found {record.get('seq')}"
            return ChainReport(False, seq - 1, prev, Break(seq, rid, gap))
        if record.get("prev_hash") != prev:
            return ChainReport(False, seq - 1, prev, Break(seq, rid, "prev_hash does not match the previous record"))
        if record_hash(record) != record.get("hash"):
            return ChainReport(False, seq - 1, prev, Break(seq, rid, "content changed: hash does not match"))
        prev = str(record["hash"])
    if expected_head is not None and expected_head != prev:
        return ChainReport(False, seq, prev, Break(seq + 1, "head", "head pointer does not match the last record"))
    return ChainReport(True, seq, prev)
