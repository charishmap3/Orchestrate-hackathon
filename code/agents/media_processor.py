from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from code.models.schemas import DataCatalog


LOGGER = logging.getLogger(__name__)


@dataclass
class MediaInformation:
    media_type: Optional[str]
    media_id: Optional[str]
    file_path: Optional[Path]
    analysis: Optional[dict[str, Any]] = None


class MediaProcessor:
    """Resolve media file paths and report missing or invalid media safely."""

    def __init__(self, catalog: DataCatalog, logger: logging.Logger | None = None) -> None:
        self.catalog = catalog
        self.logger = logger or LOGGER

    def process(self, message_id: str, message_row: Optional[pd.Series] = None) -> MediaInformation:
        self.logger.debug('Processing media for message_id=%s', message_id)
        message = None

        if message_row is not None:
            message = message_row
        else:
            found = self.catalog.messages[self.catalog.messages['message_id'] == message_id]
            if not found.empty:
                message = found.iloc[0]

        if message is None or message.empty:
            raise ValueError(f'Message not found: {message_id}')

        media_type = message.get('media_type')
        media_id = message.get('media_id')
        if pd.isna(media_type) or pd.isna(media_id):
            return MediaInformation(media_type=None, media_id=None, file_path=None)

        file_path = None
        if media_type == 'image':
            entry = self.catalog.images[self.catalog.images['image_id'] == media_id]
            if not entry.empty:
                try:
                    candidate = Path(entry.iloc[0]['file_path'])
                    file_path = candidate if candidate.is_absolute() else self.catalog.dataset_path / candidate
                except Exception as exc:
                    self.logger.warning('Invalid image metadata for message_id=%s: %s', message_id, exc)
        elif media_type == 'voice':
            entry = self.catalog.voice_notes[self.catalog.voice_notes['voice_note_id'] == media_id]
            if not entry.empty:
                try:
                    candidate = Path(entry.iloc[0]['file_path'])
                    file_path = candidate if candidate.is_absolute() else self.catalog.dataset_path / candidate
                except Exception as exc:
                    self.logger.warning('Invalid voice metadata for message_id=%s: %s', message_id, exc)

        if file_path is not None and not file_path.exists():
            self.logger.warning('Media file not found for message_id=%s: %s', message_id, file_path)
            file_path = None

        return MediaInformation(media_type=media_type, media_id=media_id, file_path=file_path)
