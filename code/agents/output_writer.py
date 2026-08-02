from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from code.agents.context_builder import build_routing_context
from code.agents.retrieval import retrieve_historical_evidence
from code.agents.router import RoutingEngine
from code.models.routing_decision import RoutingDecision
from code.models.schemas import DataCatalog


LOGGER = logging.getLogger(__name__)

ALLOWED_ACTIONS = {'notify', 'digest', 'mute'}
ALLOWED_MESSAGE_TYPES = {
    'personal',
    'urgent',
    'event',
    'payment',
    'business_update',
    'promotion',
    'greeting',
    'forward',
    'spam',
    'scam',
    'unknown',
}


class OutputWriter:
    def __init__(self, catalog: DataCatalog, logger: logging.Logger | None = None) -> None:
        self.catalog = catalog
        self.logger = logger or LOGGER
        self.router = RoutingEngine(catalog, logger=self.logger)

    def write(self, output_path: Path) -> pd.DataFrame:
        messages = self.catalog.messages.copy()
        rows: list[dict[str, Any]] = []
        total = len(messages)

        self.logger.info('Starting output generation for %s messages', total)
        for index, row in enumerate(messages.itertuples(index=False), start=1):
            message_id = getattr(row, 'message_id')
            if index % 10 == 0 or index == 1:
                self.logger.info('Processing progress: %s/%s', index, total)
            try:
                context = build_routing_context(message_id, self.catalog)
                history = retrieve_historical_evidence(message_id, self.catalog)
                decision = self.router.route(context, history)
                rows.append(self._decision_row(decision))
            except Exception as error:
                self.logger.debug('Processing failed for message_id=%s: %s', message_id, error)
                rows.append(self._fallback_row(message_id, str(error)))

        output_df = pd.DataFrame(rows, columns=['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids'])
        output_df = self._validate_and_normalize(output_df)
        output_df = output_df[['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids']]
        output_df.to_csv(output_path, index=False)
        self.logger.info('Output CSV generated')
        return output_df

    def _decision_row(self, decision: RoutingDecision) -> dict[str, Any]:
        evidence_text = 'none'
        if decision.evidence_message_ids:
            evidence_text = ';'.join(str(eid) for eid in decision.evidence_message_ids)

        return {
            'message_id': decision.message_id,
            'action': decision.action,
            'message_type': decision.message_type,
            'reason': decision.reason,
            'confidence': float(decision.confidence),
            'evidence_message_ids': evidence_text,
        }

    def _fallback_row(self, message_id: str, error_message: str) -> dict[str, Any]:
        reason = f'Processing error: {error_message}'
        if len(reason) > 120:
            reason = reason[:117] + '...'
        return {
            'message_id': message_id,
            'action': 'mute',
            'message_type': 'unknown',
            'reason': reason,
            'confidence': 0.0,
            'evidence_message_ids': 'none',
        }

    def _validate_and_normalize(self, output_df: pd.DataFrame) -> pd.DataFrame:
        if output_df['message_id'].isna().any():
            self.logger.warning('Missing message_id values found. Filling with empty string.')
            output_df['message_id'] = output_df['message_id'].fillna('')

        duplicates = output_df['message_id'][output_df['message_id'].duplicated()]
        if not duplicates.empty:
            self.logger.warning('Duplicate message_id values found: %s', duplicates.unique())

        output_df['action'] = output_df['action'].apply(self._normalize_action)
        output_df['message_type'] = output_df['message_type'].apply(self._normalize_message_type)
        output_df['confidence'] = output_df['confidence'].apply(self._normalize_confidence)
        output_df['evidence_message_ids'] = output_df['evidence_message_ids'].fillna('none').replace('', 'none')

        return output_df

    def _normalize_action(self, action: Any) -> str:
        if isinstance(action, str) and action in ALLOWED_ACTIONS:
            return action
        self.logger.warning('Invalid action value "%s" detected. Defaulting to mute.', action)
        return 'mute'

    def _normalize_message_type(self, message_type: Any) -> str:
        if isinstance(message_type, str) and message_type in ALLOWED_MESSAGE_TYPES:
            return message_type
        self.logger.warning('Invalid message_type value "%s" detected. Defaulting to unknown.', message_type)
        return 'unknown'

    def _normalize_confidence(self, confidence: Any) -> float:
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            self.logger.warning('Invalid confidence value "%s" detected. Defaulting to 0.0.', confidence)
            return 0.0

        if value < 0.0:
            self.logger.warning('Confidence below 0.0 corrected to 0.0 for value %s', value)
            return 0.0
        if value > 1.0:
            self.logger.warning('Confidence above 1.0 corrected to 1.0 for value %s', value)
            return 1.0
        return value
