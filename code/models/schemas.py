from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class DataCatalog:
    dataset_path: Path
    users: pd.DataFrame
    groups: pd.DataFrame
    group_members: pd.DataFrame
    business_accounts: pd.DataFrame
    user_business_history: pd.DataFrame
    daily_notification_summary: pd.DataFrame
    messages: pd.DataFrame
    message_history: pd.DataFrame
    message_events: pd.DataFrame
    images: pd.DataFrame
    voice_notes: pd.DataFrame
    sample_messages: pd.DataFrame
    index_cache: Optional[dict[str, dict]] = None


@dataclass(frozen=True)
class RoutingContext:
    message_id: str
    message: pd.Series
    user: Optional[pd.Series]
    sender_user: Optional[pd.Series]
    group: Optional[pd.Series]
    group_membership: Optional[pd.Series]
    group_members: pd.DataFrame
    business_account: Optional[pd.Series]
    user_business_history: pd.DataFrame
    notification_summary: pd.DataFrame


@dataclass(frozen=True)
class HistoricalRetrieval:
    message_id: str
    evidence_message_ids: List[str]
    related_messages: pd.DataFrame
    related_events: pd.DataFrame
