from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from code.agents.media_processor import MediaInformation
from code.agents.reasoner import RuleBasedReasoner
from code.models.routing_decision import RoutingDecision
from code.models.schemas import DataCatalog, HistoricalRetrieval, RoutingContext
from code.services.gemini_service import GeminiService
from code.utils.prompt_builder import build_gemini_prompt

LOGGER = logging.getLogger(__name__)


class GeminiReasoner:
    def __init__(self, catalog: DataCatalog, logger: logging.Logger | None = None) -> None:
        self.catalog = catalog
        self.logger = logger or LOGGER
        self.gemini_service = GeminiService(logger=self.logger)
        self.rule_reasoner = RuleBasedReasoner(logger=self.logger)
        self.gemini_decisions = 0
        self.rule_decisions = 0
        self.fallback_count = 0

    def reason(self, context: RoutingContext, history: HistoricalRetrieval, media_info: MediaInformation, message_row: Optional[pd.Series] = None) -> RoutingDecision:
        self.logger.info('Hybrid reasoning over message_id=%s', context.message_id)

        if media_info.media_type in {'image', 'voice'}:
            try:
                media_analysis = self._analyze_media(context, media_info)
                media_info.analysis = media_analysis
                self.logger.info('Media analysis completed for message_id=%s: %s', context.message_id, media_analysis)
            except Exception as error:
                self.fallback_count += 1
                self.logger.warning('Media analysis failed for message_id=%s, falling back to rule-based reasoner: %s', context.message_id, error)
                return self.rule_reasoner.reason(context, history, media_info)

            return self.rule_reasoner.reason(context, history, media_info)

        rule_decision = self.rule_reasoner.reason(context, history, media_info)
        if self._use_rule_fallback(context, rule_decision, media_info):
            self.rule_decisions += 1
            self.logger.info('Rule-based fallback triggered for message_id=%s', context.message_id)
            return rule_decision

        prompt = self._build_prompt(context, history, media_info, message_row)
        try:
            start = time.perf_counter()
            response_text = self.gemini_service.generate(prompt, media_path=media_info.file_path)
            latency = time.perf_counter() - start
            self.logger.info('Gemini request latency: %.3fs for message_id=%s', latency, context.message_id)
            self.logger.debug('Gemini request prompt: %s', prompt)
            self.logger.debug('Gemini response raw: %s', response_text)

            gemini_json = self._parse_response(response_text)
            decision = self._build_decision(context, history, gemini_json)
            self.gemini_decisions += 1
            return decision
        except Exception as error:
            self.fallback_count += 1
            self.logger.warning('Gemini failed for message_id=%s, falling back to rule-based reasoner: %s', context.message_id, error)
            self.rule_decisions += 1
            return rule_decision

    def _use_rule_fallback(self, context: RoutingContext, rule_decision: RoutingDecision, media_info: MediaInformation) -> bool:
        message = context.message
        if context.business_account is not None and int(context.business_account.get('verified') or 0) == 1:
            if context.business_account.get('category') in {'bank', 'finance', 'insurance', 'payment'}:
                return True
        if context.group_membership is not None and int(context.group_membership.get('group_muted_by_user') or 0) == 1:
            return True
        if int(message.get('forwarded_count') or 0) > 3:
            return True
        if rule_decision.action == 'mute' and rule_decision.message_type == 'spam':
            return True
        text = str(message.get('message_text') or '').lower()
        if any(token in text for token in ['otp', 'password', 'verify', 'account blocked', 'urgent action']):
            return True
        return False

    def _build_prompt(self, context: RoutingContext, history: HistoricalRetrieval, media_info: MediaInformation, message_row: Optional[pd.Series]) -> str:
        message = context.message.to_dict()
        user = context.user.to_dict() if context.user is not None else {}
        group = context.group.to_dict() if context.group is not None else None
        business = context.business_account.to_dict() if context.business_account is not None else None
        notification_summary = context.notification_summary.fillna('').to_dict(orient='records')
        historical_messages = history.related_messages.fillna('').to_dict(orient='records')
        historical_events = history.related_events.fillna('').to_dict(orient='records')
        evidence = history.evidence_message_ids
        media_summary = self._describe_media(media_info)

        build_info = {
            'media_type': media_info.media_type,
            'media_id': media_info.media_id,
            'media_summary': media_summary,
        }
        if media_info.analysis:
            build_info['media_analysis'] = media_info.analysis

        return build_gemini_prompt(
            message=message,
            user=user,
            group=group,
            business=business,
            notification_summary=notification_summary,
            historical_messages=historical_messages,
            historical_events=historical_events,
            evidence=evidence,
            media_info=build_info,
        )

    def _analyze_media(self, context: RoutingContext, media_info: MediaInformation) -> dict[str, Any]:
        prompt = self._build_media_prompt(context, media_info)
        self.logger.info('Running media analysis for message_id=%s media_type=%s', context.message_id, media_info.media_type)
        response_text = self.gemini_service.generate(prompt, media_path=media_info.file_path)
        self.logger.debug('Gemini media analysis response raw: %s', response_text)
        parsed = self.gemini_service.parse_json(response_text)

        if not isinstance(parsed, dict):
            raise ValueError('Gemini media analysis response must be JSON object.')

        if media_info.media_type == 'image':
            return {
                'summary': str(parsed.get('summary', '')).strip(),
                'category': str(parsed.get('category', '')).strip(),
                'urgency': str(parsed.get('urgency', '')).strip(),
                'risk': str(parsed.get('risk', '')).strip(),
                'confidence': float(parsed.get('confidence', 0.0)),
            }

        return {
            'summary': str(parsed.get('summary', '')).strip(),
            'urgency': str(parsed.get('urgency', '')).strip(),
            'reminder': str(parsed.get('reminder', '')).strip(),
            'payment_request': str(parsed.get('payment_request', '')).strip(),
            'emergency': str(parsed.get('emergency', '')).strip(),
            'scam': str(parsed.get('scam', '')).strip(),
            'confidence': float(parsed.get('confidence', 0.0)),
        }

    def _build_media_prompt(self, context: RoutingContext, media_info: MediaInformation) -> str:
        if media_info.media_type == 'image':
            instructions = [
                'You are analyzing an image attached to a WhatsApp message.',
                'Describe the image and detect whether it contains posters, payment reminders, event notices, advertisements, scams, or urgency.',
                'Return only valid JSON with the fields: summary, category, urgency, risk, confidence.',
                'confidence should be a float between 0.0 and 1.0.',
            ]
        else:
            instructions = [
                'You are analyzing a voice note attached to a WhatsApp message.',
                'Transcribe or summarize the audio if possible, and extract urgency, reminders, payment requests, emergency signals, and scams.',
                'Return only valid JSON with the fields: summary, urgency, reminder, payment_request, emergency, scam, confidence.',
                'confidence should be a float between 0.0 and 1.0.',
            ]

        text = str(context.message.get('message_text') or '')
        prompt_lines = [
            *instructions,
            '',
            'Message text:',
            f'"""{text}"""',
            '',
            'Attached media path will be provided to Gemini as an additional multimodal input.',
            '',
            'Use only the supplied context.',
            '',
            'Example image output:',
            '{',
            '  "summary":"A poster advertising a payment reminder for a utility bill.",',
            '  "category":"payment reminder",',
            '  "urgency":"medium",',
            '  "risk":"medium",',
            '  "confidence":0.88',
            '}',
            '',
            'Example voice output:',
            '{',
            '  "summary":"Caller requests immediate payment and mentions an overdue invoice.",',
            '  "urgency":"high",',
            '  "reminder":"payment due",',
            '  "payment_request":"yes",',
            '  "emergency":"no",',
            '  "scam":"possible scam",',
            '  "confidence":0.90',
            '}',
        ]

        return '\n'.join(prompt_lines)

    def _describe_media(self, media_info: MediaInformation) -> str:
        if media_info.media_type == 'image' and media_info.file_path is not None:
            return (
                'Image attached: analyze the visual scene for posters, payment reminders, event notices, advertisements, scams, urgency, and any other user-facing signal.'
            )
        if media_info.media_type == 'voice' and media_info.file_path is not None:
            return (
                'Voice note attached: transcribe or summarize the audio if possible, then identify urgency, payment requests, scams, reminders, or other actionable cues.'
            )
        return 'No media attached.'

    def _parse_response(self, text: str) -> dict[str, Any]:
        parsed = self.gemini_service.parse_json(text)
        if not isinstance(parsed, dict):
            raise ValueError('Gemini response JSON must be an object.')
        return parsed

    def _build_decision(self, context: RoutingContext, history: HistoricalRetrieval, gemini_json: dict[str, Any]) -> RoutingDecision:
        action = str(gemini_json.get('action', '')).strip()
        message_type = str(gemini_json.get('message_type', '')).strip()
        reason = str(gemini_json.get('reason', '')).strip()
        confidence = float(gemini_json.get('confidence', 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
        if action not in {'notify', 'digest', 'mute'}:
            raise ValueError(f'Invalid action from Gemini: {action}')
        if not message_type:
            raise ValueError('Gemini response missing message_type')
        if not reason:
            raise ValueError('Gemini response missing reason')

        self.logger.info('Gemini decision for %s: %s / %s / %.2f', context.message_id, action, message_type, confidence)
        return RoutingDecision(
            message_id=context.message_id,
            action=action,
            message_type=message_type,
            reason=reason,
            confidence=confidence,
            evidence_message_ids=history.evidence_message_ids,
        )
