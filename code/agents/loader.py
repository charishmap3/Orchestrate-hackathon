from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from code.models.schemas import DataCatalog


LOGGER = logging.getLogger(__name__)


class DatasetLoader:
    """Load the dataset artifacts and prepare lightweight lookup caches."""
    REQUIRED_FILES: Dict[str, str] = {
        'business_accounts.csv': 'business_accounts',
        'daily_notification_summary.csv': 'daily_notification_summary',
        'groups.csv': 'groups',
        'group_members.csv': 'group_members',
        'images.csv': 'images',
        'message_events.csv': 'message_events',
        'message_history.csv': 'message_history',
        'messages.csv': 'messages',
        'sample_messages.csv': 'sample_messages',
        'user_business_history.csv': 'user_business_history',
        'users.csv': 'users',
        'voice_notes.csv': 'voice_notes',
    }

    def __init__(self, dataset_path: Path, logger: logging.Logger | None = None) -> None:
        self.dataset_path = dataset_path
        self.logger = logger or LOGGER

    def load_all(self) -> DataCatalog:
        self.logger.info('Loading dataset from %s', self.dataset_path)
        self._validate_dataset_path()
        loaded: Dict[str, pd.DataFrame] = {}

        for filename, attr_name in self.REQUIRED_FILES.items():
            path = self.dataset_path / filename
            self.logger.debug('Reading %s', path)
            loaded[attr_name] = self._load_csv(path)

        index_cache = self._build_index_cache(loaded)
        self.logger.info('Dataset loaded')
        return DataCatalog(dataset_path=self.dataset_path, **loaded, index_cache=index_cache)

    def _validate_dataset_path(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f'Dataset directory does not exist: {self.dataset_path}')

        missing = [name for name in self.REQUIRED_FILES if not (self.dataset_path / name).exists()]
        if missing:
            raise FileNotFoundError(f'Missing dataset files: {missing}')

    def _load_csv(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.logger.error('Failed to read %s: %s', path, exc)
            raise

        required_columns = self._required_columns_for(path.name)
        for column_name in required_columns:
            if column_name not in df.columns:
                self.logger.warning('Missing column %s in %s; creating empty values.', column_name, path.name)
                df[column_name] = pd.NA

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        if 'joined_at' in df.columns:
            df['joined_at'] = pd.to_datetime(df['joined_at'], errors='coerce')
        if 'last_activity_at' in df.columns:
            df['last_activity_at'] = pd.to_datetime(df['last_activity_at'], errors='coerce')
        if 'last_reply_at' in df.columns:
            df['last_reply_at'] = pd.to_datetime(df['last_reply_at'], errors='coerce')

        return df

    def _required_columns_for(self, filename: str) -> list[str]:
        required_columns: dict[str, list[str]] = {
            'business_accounts.csv': ['business_id', 'verified', 'category', 'official_domain', 'domain_used_by_sender', 'user_reports_30d', 'account_age_days'],
            'daily_notification_summary.csv': ['user_id', 'notifications_sent', 'notifications_dismissed', 'date'],
            'groups.csv': ['group_id', 'group_type'],
            'group_members.csv': ['group_id', 'user_id', 'group_admin', 'group_muted_by_user'],
            'images.csv': ['image_id', 'file_path'],
            'message_events.csv': ['message_id', 'message_opened', 'message_replied', 'notification_dismissed', 'message_reported', 'muted_after_message'],
            'message_history.csv': ['message_id', 'user_id', 'sender_user_id', 'business_id', 'group_id', 'conversation_type', 'message_text', 'created_at'],
            'messages.csv': ['message_id', 'user_id', 'conversation_type', 'message_text', 'media_type', 'media_id', 'forwarded_count', 'group_id', 'business_id', 'sender_user_id'],
            'sample_messages.csv': ['message_id', 'user_id', 'conversation_type', 'group_id', 'business_id', 'sender_user_id', 'created_at', 'message_text', 'media_type', 'media_id', 'forwarded_count', 'action', 'message_type'],
            'user_business_history.csv': ['user_id', 'business_id', 'messages_opened_30d', 'messages_dismissed_30d', 'messages_replied_30d', 'activity_count_180d', 'last_activity_at'],
            'users.csv': ['user_id', 'messages_replied_30d', 'messages_opened_30d'],
            'voice_notes.csv': ['voice_note_id', 'file_path'],
        }
        return required_columns.get(filename, [])

    def _build_index_cache(self, loaded: Dict[str, pd.DataFrame]) -> dict[str, dict]:
        cache: dict[str, dict] = {}
        for table_name in ['messages', 'users', 'groups', 'business_accounts']:
            dataframe = loaded.get(table_name)
            if dataframe is None or dataframe.empty:
                continue
            key_column = self._primary_key_for(table_name)
            if key_column is None or key_column not in dataframe.columns:
                continue
            cache[table_name] = {
                str(row.get(key_column)): row for _, row in dataframe.iterrows() if pd.notna(row.get(key_column))
            }
        return cache

    def _primary_key_for(self, table_name: str) -> str | None:
        mapping = {
            'messages': 'message_id',
            'users': 'user_id',
            'groups': 'group_id',
            'business_accounts': 'business_id',
        }
        return mapping.get(table_name)
