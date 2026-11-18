"""Versioned, immutable material handoffs shared by long-lived agents."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
from threading import RLock


class HandoffError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterialRef:
    name: str
    version: int
    sha256: str

    def to_dict(self):
        return asdict(self)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class MaterialBoard:
    """Single writer per material; consumers receive copies of pinned versions."""

    def __init__(self):
        self._lock = RLock()
        self._items = {}

    def publish(self, name: str, payload: dict, expected_version: int | None = None) -> MaterialRef:
        with self._lock:
            previous = self._items.get(name)
            version = previous[0].version if previous else 0
            if expected_version is not None and expected_version != version:
                raise HandoffError(f"材料 {name} 版本冲突，预期 {expected_version}，实际 {version}")
            frozen = deepcopy(payload)
            ref = MaterialRef(name, version + 1, _digest(frozen))
            self._items[name] = ref, frozen
            return ref

    def refs(self, *names: str) -> list[MaterialRef]:
        with self._lock:
            return [self._items[name][0] for name in names if name in self._items]

    def read(self, reference: MaterialRef) -> dict:
        with self._lock:
            current = self._items.get(reference.name)
            if current is None or current[0] != reference:
                raise HandoffError(f"材料 {reference.name} 的交接版本已失效，须重新读取")
            if _digest(current[1]) != reference.sha256:
                raise HandoffError(f"材料 {reference.name} 的内容与交接哈希不符")
            return deepcopy(current[1])

    def validate(self, references: list[MaterialRef]) -> None:
        for ref in references:
            self.read(ref)

    def manifest(self):
        with self._lock:
            return {name: reference.to_dict() for name, (reference, _) in self._items.items()}
