from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from code.models.schemas import DataCatalog, RoutingContext


LOGGER = logging.getLogger(__name__)


def _safe_locate_one(frame: pd.DataFrame, **criteria) -> Optional[pd.Series]:
    if frame.empty:
        return None
    filtered = frame.copy()
    for key, value in criteria.items():
        if pd.isna(value):
            filtered = filtered[filtered[key].isna()]
        else:
            filtered = filtered[filtered[key] == value]
    if filtered.empty:
        return None
    return filtered.iloc[0]


def build_routing_context(message_id: str, catalog: DataCatalog, logger: logging.Logger | None = None) -> RoutingContext:
    """Build a routing context for a message using the loaded dataset tables."""
    logger = logger or LOGGER
    logger.debug('Building routing context for message_id=%s', message_id)

    message = _safe_locate_one(catalog.messages, message_id=message_id)
    if message is None:
        raise ValueError(f'Message not found in messages.csv: {message_id}')

    return _build_routing_context_from_row(message, catalog, logger)


def build_routing_context_for_row(message: pd.Series, catalog: DataCatalog, logger: logging.Logger | None = None) -> RoutingContext:
    """Build a routing context directly from a sample message row."""
    logger = logger or LOGGER
    logger.debug('Building routing context for sample message_id=%s', message.get('message_id'))
    return _build_routing_context_from_row(message, catalog, logger)


def _build_routing_context_from_row(message: pd.Series, catalog: DataCatalog, logger: logging.Logger | None = None) -> RoutingContext:
    logger = logger or LOGGER
    user = _safe_locate_one(catalog.users, user_id=message['user_id'])
    sender_user = None
    if pd.notna(message.get('sender_user_id')):
        sender_user = _safe_locate_one(catalog.users, user_id=message['sender_user_id'])

    group = None
    group_membership = None
    group_members = pd.DataFrame(columns=catalog.group_members.columns)
    if message.get('conversation_type') == 'group' and pd.notna(message.get('group_id')):
        group = _safe_locate_one(catalog.groups, group_id=message['group_id'])
        group_members = catalog.group_members[catalog.group_members['group_id'] == message['group_id']].copy()
        if user is not None:
            group_membership = _safe_locate_one(catalog.group_members, group_id=message['group_id'], user_id=message['user_id'])

    business_account = None
    user_business_history = pd.DataFrame(columns=catalog.user_business_history.columns)
    if pd.notna(message.get('business_id')):
        business_account = _safe_locate_one(catalog.business_accounts, business_id=message['business_id'])
        if user is not None:
            user_business_history = catalog.user_business_history[
                (catalog.user_business_history['user_id'] == message['user_id'])
                & (catalog.user_business_history['business_id'] == message['business_id'])
            ].copy()

    notification_summary = catalog.daily_notification_summary[
        catalog.daily_notification_summary['user_id'] == message['user_id']
    ].sort_values(by='date', ascending=False).copy()

    return RoutingContext(
        message_id=str(message['message_id']),
        message=message,
        user=user,
        sender_user=sender_user,
        group=group,
        group_membership=group_membership,
        group_members=group_members,
        business_account=business_account,
        user_business_history=user_business_history,
        notification_summary=notification_summary,
    )
