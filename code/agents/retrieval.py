from __future__ import annotations

import logging
from typing import List

import pandas as pd

from code.models.schemas import DataCatalog, HistoricalRetrieval


LOGGER = logging.getLogger(__name__)


def _match_by_metadata(message: pd.Series, history: pd.DataFrame) -> pd.DataFrame:
    if message.get('conversation_type') == 'business' and pd.notna(message.get('business_id')):
        return history[history['business_id'] == message['business_id']]

    if message.get('conversation_type') == 'group' and pd.notna(message.get('group_id')):
        return history[history['group_id'] == message['group_id']]

    matched = pd.DataFrame()
    if pd.notna(message.get('sender_user_id')):
        matched = history[history['sender_user_id'] == message['sender_user_id']]
    if matched.empty and pd.notna(message.get('user_id')):
        matched = history[history['user_id'] == message['user_id']]
    return matched


def retrieve_historical_evidence(message_id: str, catalog: DataCatalog, logger: logging.Logger | None = None) -> HistoricalRetrieval:
    """Retrieve historical evidence for a message using the existing dataset metadata."""
    logger = logger or LOGGER
    logger.debug('Retrieving historical evidence for message_id=%s', message_id)

    message = catalog.messages[catalog.messages['message_id'] == message_id]
    if message.empty:
        raise ValueError(f'Message not found in messages.csv: {message_id}')
    message = message.iloc[0]

    return retrieve_historical_evidence_for_row(message_id, message, catalog, logger)


def retrieve_historical_evidence_for_row(message_id: str, message: pd.Series, catalog: DataCatalog, logger: logging.Logger | None = None) -> HistoricalRetrieval:
    logger = logger or LOGGER
    matched_history = _match_by_metadata(message, catalog.message_history)
    if matched_history.empty:
        logger.debug('No historical metadata match found for %s, falling back to conversation type only', message_id)
        matched_history = catalog.message_history[catalog.message_history['conversation_type'] == message['conversation_type']]

    related_events = catalog.message_events[catalog.message_events['message_id'].isin(matched_history['message_id'])].copy()
    related_messages = matched_history.sort_values(by='created_at', ascending=False).head(10).copy()
    evidence_message_ids: List[str] = related_messages['message_id'].astype(str).tolist()

    return HistoricalRetrieval(
        message_id=message_id,
        evidence_message_ids=evidence_message_ids,
        related_messages=related_messages,
        related_events=related_events,
    )
