from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SftRecord:
    id: str
    instruction: str
    input: str
    output: str
    task_type: str
    source: str
    source_file: str
    row_id: int
    group_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DpoRecord:
    prompt: str
    chosen: str
    rejected: str
    source: str
    quality_tag: str
    source_id: str

    def to_dict(self) -> dict:
        return asdict(self)
