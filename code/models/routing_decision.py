from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RoutingDecision:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: List[str]
