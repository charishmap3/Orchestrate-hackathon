from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from code.agents.gemini_reasoner import GeminiReasoner
from code.agents.media_processor import MediaProcessor, MediaInformation
from code.agents.reasoner import RuleBasedReasoner
from code.models.routing_decision import RoutingDecision
from code.models.schemas import DataCatalog, HistoricalRetrieval, RoutingContext


LOGGER = logging.getLogger(__name__)


class RoutingEngine:
    def __init__(self, catalog: DataCatalog, logger: logging.Logger | None = None) -> None:
        self.catalog = catalog
        self.logger = logger or LOGGER
        self.rule_reasoner = RuleBasedReasoner(logger=self.logger)
        self.gemini_reasoner = GeminiReasoner(catalog, logger=self.logger)
        self.media_processor = MediaProcessor(catalog, logger=self.logger)
        self.gemini_calls = 0
        self.rule_calls = 0
        self.images_processed = 0
        self.voice_notes_processed = 0

    def route(self, context: RoutingContext, history: HistoricalRetrieval, message_row: Optional[pd.Series] = None) -> RoutingDecision:
        self.logger.debug('Routing message_id=%s', context.message_id)
        media_info = self.media_processor.process(context.message_id, message_row=message_row)

        if media_info.media_type == 'image':
            self.images_processed += 1
        elif media_info.media_type == 'voice':
            self.voice_notes_processed += 1

        if media_info.media_type in {'image', 'voice'}:
            if media_info.file_path is None:
                self.logger.debug('Media file missing for message_id=%s, using rule-based reasoner.', context.message_id)
                self.rule_calls += 1
                return self.rule_reasoner.reason(context, history, media_info)

            if self.gemini_reasoner.gemini_service.is_available():
                self.gemini_calls += 1
                return self.gemini_reasoner.reason(context, history, media_info, message_row=message_row)

            self.logger.debug('Gemini service unavailable, using rule-based reasoner for %s', context.message_id)
            self.rule_calls += 1
            return self.rule_reasoner.reason(context, history, media_info)

        rule_decision = self.rule_reasoner.reason(context, history, media_info)
        if self._should_use_gemini(context, rule_decision, media_info):
            if self.gemini_reasoner.gemini_service.is_available():
                self.gemini_calls += 1
                return self.gemini_reasoner.reason(context, history, media_info, message_row=message_row)
            self.logger.debug('Gemini service unavailable, using rule-based reasoner for %s', context.message_id)
            self.rule_calls += 1
            return rule_decision

        self.rule_calls += 1
        return rule_decision

    def _should_use_gemini(
        self,
        context: RoutingContext,
        rule_decision: RoutingDecision,
        media_info: MediaInformation,
    ) -> bool:
        if media_info.media_type in {'image', 'voice'}:
            return True

        if rule_decision.confidence < 0.45:
            return True

        text = str(context.message.get('message_text') or '').lower()
        verified_business = (
            context.business_account is not None
            and int(context.business_account.get('verified') or 0) == 1
        )
        forwarded_count = int(context.message.get('forwarded_count') or 0)

        ambiguous_signals = (
            rule_decision.action == 'notify'
            and rule_decision.message_type in {'payment', 'urgent', 'personal_alert'}
            and not verified_business
            and rule_decision.confidence < 0.65
        )
        if ambiguous_signals:
            return True

        if forwarded_count > 2 and rule_decision.confidence < 0.5:
            return True

        suspicious_tokens = {
            'urgent', 'payment', 'invoice', 'due', 'reminder', 'verify', 'account', 'password',
            'scam', 'offer', 'promotion', 'advertisement', 'event', 'invitation', 'alert',
            'security', 'blocked', 'transfer', 'refund', 'lottery', 'win', 'free', 'subscription',
            'otp', 'pin', 'urgent action', 'verify now', 'account blocked', 'overdue', 'late fee',
        }
        conflicting_signals = any(token in text for token in suspicious_tokens) and rule_decision.action != 'notify'
        if conflicting_signals and rule_decision.confidence < 0.55:
            return True

        return False
