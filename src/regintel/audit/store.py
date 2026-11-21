"""Where audit records live: in memory (development, tests) or Cosmos DB (Azure).

Cosmos layout: container `audit`, partition key `/day` (YYYY-MM-DD, UTC), no TTL. Each day's
partition holds that day's records (type "answer") and one head document (type "head") with the
last sequence number and hash. An append writes the record and moves the head in one
transactional batch, conditional on the head's ETag: two replicas appending at once cannot both
take the same sequence number, so the chain cannot fork. The loser retries on the new head.

Records are never updated or deleted by the application; the API identity can write, but nothing
in the code path calls replace or delete on a record.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from regintel.audit.chain import GENESIS, ChainReport, seal, verify

HEAD_ID = "head"
APPEND_ATTEMPTS = 8


class AuditWriteError(Exception):
    """The record could not be stored; the answer must not be released without it."""


class AuditStore(Protocol):
    def append(self, record: dict[str, Any]) -> dict[str, Any]: ...
    def get(self, record_id: str) -> dict[str, Any] | None: ...
    def days(self) -> list[tuple[str, str]]: ...  # (day, head hash), oldest first
    def records(self, day: str) -> list[dict[str, Any]]: ...


def verify_store(store: AuditStore) -> dict[str, Any]:
    """Recompute the whole chain, day by day, each day starting from the previous day's head.
    Returns the first broken link, with its day."""
    prev, total = GENESIS, 0
    for day, head in store.days():
        report: ChainReport = verify(store.records(day), prev, head)
        total += report.checked
        if not report.ok:
            return {"ok": False, "records_checked": total, "day": day, "first_break": report.as_dict()["first_break"]}
        prev = report.head
    return {"ok": True, "records_checked": total, "day": None, "first_break": None, "head": prev}


class InMemoryAuditStore:
    def __init__(self) -> None:
        self._days: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            day = record["day"]
            if day in self._days:
                chain = self._days[day]
                seq, prev = len(chain) + 1, chain[-1]["hash"]
            else:
                earlier = [d for d in self._days if d < day]
                seq, prev = 1, (self._days[max(earlier)][-1]["hash"] if earlier else GENESIS)
            sealed = seal(record, seq, prev)
            self._days.setdefault(day, []).append(sealed)
            return dict(sealed)

    def get(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            return next((dict(r) for chain in self._days.values() for r in chain if r["id"] == record_id), None)

    def days(self) -> list[tuple[str, str]]:
        with self._lock:
            return [(day, chain[-1]["hash"]) for day, chain in sorted(self._days.items())]

    def records(self, day: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._days.get(day, [])]

    def tamper(self, day: str, seq: int, **changes: Any) -> None:
        """Test helper: edit a stored record in place, as someone with database access could."""
        with self._lock:
            self._days[day][seq - 1].update(changes)


class CosmosAuditStore:
    def __init__(self, container: Any) -> None:
        self.container = container  # azure.cosmos ContainerProxy

    @classmethod
    def connect(
        cls, endpoint: str, credential: Any, database: str = "regintel", container: str = "audit"
    ) -> CosmosAuditStore:
        from azure.cosmos import CosmosClient

        client = CosmosClient(endpoint, credential=credential)
        return cls(client.get_database_client(database).get_container_client(container))

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        from azure.cosmos.exceptions import CosmosBatchOperationError, CosmosResourceNotFoundError

        day = record["day"]
        for _ in range(APPEND_ATTEMPTS):
            try:
                head = self.container.read_item(HEAD_ID, partition_key=day)
                seq, prev = int(head["seq"]) + 1, str(head["hash"])
                condition: dict[str, Any] = {"if_match_etag": head["_etag"]}
            except CosmosResourceNotFoundError:
                head, seq, prev, condition = None, 1, self._previous_head(day), {}
            sealed = seal(record, seq, prev)
            new_head = {
                "id": HEAD_ID,
                "day": day,
                "type": "head",
                "seq": seq,
                "hash": sealed["hash"],
                "last": sealed["id"],
            }
            move_head = ("replace", (HEAD_ID, new_head), condition) if head else ("create", (new_head,))
            try:
                self.container.execute_item_batch([("create", (sealed,)), move_head], partition_key=day)
                return sealed
            except CosmosBatchOperationError as error:
                statuses = {int(r.get("statusCode", 0)) for r in error.operation_responses or []}
                if statuses & {409, 412}:  # another replica moved the head first: retry on the new one
                    continue
                raise AuditWriteError(str(error)) from error
        raise AuditWriteError(f"could not append after {APPEND_ATTEMPTS} attempts (contention on {day})")

    def _previous_head(self, day: str) -> str:
        rows = list(self.container.query_items(
            "SELECT TOP 1 c.hash FROM c WHERE c.type = 'head' AND c.day < @day ORDER BY c.day DESC",
            parameters=[{"name": "@day", "value": day}], enable_cross_partition_query=True,
        ))  # fmt: skip
        return str(rows[0]["hash"]) if rows else GENESIS

    def get(self, record_id: str) -> dict[str, Any] | None:
        rows = list(self.container.query_items(
            "SELECT * FROM c WHERE c.type = 'answer' AND c.id = @id",
            parameters=[{"name": "@id", "value": record_id}], enable_cross_partition_query=True,
        ))  # fmt: skip
        return rows[0] if rows else None

    def days(self) -> list[tuple[str, str]]:
        rows = self.container.query_items(
            "SELECT c.day, c.hash FROM c WHERE c.type = 'head'", enable_cross_partition_query=True
        )
        return sorted((str(r["day"]), str(r["hash"])) for r in rows)

    def records(self, day: str) -> list[dict[str, Any]]:
        return list(self.container.query_items("SELECT * FROM c WHERE c.type = 'answer'", partition_key=day))
