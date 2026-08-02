from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from code.agents.media_processor import MediaInformation
from code.models.routing_decision import RoutingDecision
from code.models.schemas import HistoricalRetrieval, RoutingContext


LOGGER = logging.getLogger(__name__)


def _coerce_int(value, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'y'}:
            return True
        if normalized in {'0', 'false', 'no', 'n'}:
            return False
    return default


class RuleBasedReasoner:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER

    def reason(self, context: RoutingContext, history: HistoricalRetrieval, media_info: Optional[MediaInformation] = None) -> RoutingDecision:
        self.logger.info('Reasoning over message_id=%s', context.message_id)
        action = self._decide_action(context, history, media_info)
        message_type = self._decide_message_type(context, history, media_info)
        reason = self._decide_reason(context, history, media_info, action, message_type)
        confidence = self._decide_confidence(context, history, media_info, action, message_type)

        return RoutingDecision(
            message_id=context.message_id,
            action=action,
            message_type=message_type,
            reason=reason,
            confidence=confidence,
            evidence_message_ids=history.evidence_message_ids,
        )

    def _decide_action(self, context: RoutingContext, history: HistoricalRetrieval, media_info: Optional[dict]) -> str:
        message = context.message
        conversation_type = message.get('conversation_type')
        business = context.business_account
        forwarded_count = _coerce_int(message.get('forwarded_count'), 0)
        user = context.user
        notification_summary = context.notification_summary
        group_membership = context.group_membership
        text = str(message.get('message_text') or '').lower()

        trust = self._sender_trust_score(context)
        history_score = self._historical_behavior_score(context, history)
        negated_urgency = self._has_negated_urgency(text)
        urgency = self._is_urgent(text, conversation_type) and not negated_urgency
        scam = self._is_scam(text, business)
        repeated_ads = any(token in text for token in ['offer', 'sale', 'discount', 'promo', 'newsletter', 'subscribe'])
        direct_mention_in_muted_group = (
            conversation_type == 'group'
            and group_membership is not None
            and _coerce_int(group_membership.get('group_muted_by_user'), 0) == 1
            and any(token in text for token in ['@', ' you ', 'your ', 'please respond', 'attention', 'urgent'])
        )

        media_category = self._media_analysis_field(media_info, 'category')
        media_urgency = self._media_analysis_field(media_info, 'urgency')
        media_summary = self._media_analysis_field(media_info, 'summary')
        media_confidence = self._media_analysis_confidence(media_info)
        media_scam = self._media_is_scam_media(media_info)
        media_payment = self._media_is_payment_media(media_info)
        media_emergency = self._media_is_emergency_media(media_info)
        media_invitation = self._media_is_invitation_media(media_info)
        media_delivery = self._media_is_delivery_media(media_info)
        media_ad = self._media_is_advertisement_media(media_info)
        media_school = self._media_is_school_media(media_info)

        if media_info is not None and media_info.media_type == 'image' and media_confidence >= 0.1:
            if media_scam:
                return 'mute'
            if media_emergency or media_urgency in {'high', 'urgent'}:
                return 'notify'
            if media_payment:
                return 'notify' if trust >= 0.4 else 'digest'
            if media_delivery:
                return 'notify' if trust >= 0.5 else 'digest'
            if media_invitation or media_school:
                return 'notify'
            if media_ad:
                return 'digest' if trust >= 0.3 else 'mute'
            return 'digest'

        if media_info is not None and media_info.media_type == 'voice' and media_confidence >= 0.1:
            if media_scam:
                return 'mute'
            if media_emergency or media_urgency in {'high', 'urgent'}:
                return 'notify'
            if media_payment or self._is_voice_payment_request(media_info):
                return 'notify'
            if self._is_voice_reminder(media_info):
                return 'notify'
            if media_urgency in {'medium'}:
                return 'notify'
            return 'digest'

        if conversation_type == 'business' and business is not None:
            category = str(business.get('category') or '').lower()
            if self._is_otp(text) or 'otp' in text or 'one time' in text:
                return 'notify'
            if category in {'bank', 'insurance', 'finance', 'payment'}:
                if scam or forwarded_count > 2:
                    return 'mute' if trust < 0.4 else 'notify'
                return 'notify'
            if any(token in text for token in ['school', 'campus', 'homework', 'results', 'tuition']):
                return 'notify'
            if any(token in text for token in ['emergency', 'alert', 'incident', 'safety']):
                return 'notify'
            if any(token in text for token in ['delivery', 'shipment', 'tracking', 'package']):
                return 'digest' if trust < 0.5 else 'notify'
            if repeated_ads or category in {'shopping', 'retail', 'promotions', 'fashion'}:
                if history_score < 0:
                    return 'mute'
                return 'digest'
            if scam or forwarded_count > 3 or trust < 0.25:
                return 'mute'
            if history_score > 0.1 or trust > 0.6:
                return 'notify'
            return 'digest'

        if conversation_type == 'group':
            if group_membership is not None and _coerce_int(group_membership.get('group_muted_by_user'), 0) == 1:
                if direct_mention_in_muted_group or urgency:
                    return 'notify'
                return 'mute'
            if urgency or direct_mention_in_muted_group:
                return 'notify'
            if repeated_ads or scam or forwarded_count > 2:
                return 'mute'
            return 'digest' if history_score < 0 or trust < 0.4 else 'notify'

        if conversation_type == 'personal':
            if self._is_otp(text) or urgency:
                return 'notify'
            if scam or forwarded_count > 2:
                return 'mute'
            if any(token in text for token in ['hi', 'hello', 'hey', 'good morning', 'good evening']):
                return 'digest'
            if repeated_ads or 'newsletter' in text:
                return 'digest'
            return 'notify' if history_score > 0 or trust > 0.5 else 'digest'

        if scam or forwarded_count > 3 or trust < 0.2:
            return 'mute'

        return 'digest'

    def _decide_message_type(self, context: RoutingContext, history: HistoricalRetrieval, media_info: Optional[MediaInformation]) -> str:
        message = context.message
        business = context.business_account
        forwarded_count = _coerce_int(message.get('forwarded_count'), 0)
        text = str(message.get('message_text') or '').lower()
        conversation_type = str(message.get('conversation_type') or '').lower()
        business_category = str(business.get('category') or '').lower() if business is not None else ''
        business_verified = business is not None and _coerce_int(business.get('verified'), 0) == 1
        group = context.group
        group_type = str(group.get('group_type') or '').lower() if group is not None else ''
        group_muted = context.group_membership is not None and _coerce_int(context.group_membership.get('group_muted_by_user'), 0) == 1
        history_hint = self._history_hint(history)
        trust = self._sender_trust_score(context)
        scores = {label: 0 for label in ['personal', 'urgent', 'event', 'payment', 'business_update', 'promotion', 'greeting', 'forward', 'spam', 'scam', 'unknown']}

        if self._is_scam(text, business, media_info):
            scores['scam'] += 10

        if media_info is not None and media_info.analysis is not None:
            if media_info.media_type == 'image':
                if self._media_is_scam_media(media_info):
                    scores['scam'] += 8
                if self._media_is_payment_media(media_info):
                    scores['payment'] += 7
                if self._media_is_emergency_media(media_info):
                    scores['urgent'] += 7
                if self._media_is_invitation_media(media_info) or self._media_is_school_media(media_info):
                    scores['event'] += 6
                if self._media_is_delivery_media(media_info):
                    scores['business_update'] += 6
                if self._media_is_advertisement_media(media_info):
                    scores['promotion'] += 6
            if media_info.media_type == 'voice':
                if self._media_is_scam_media(media_info):
                    scores['scam'] += 8
                if self._media_is_emergency_media(media_info):
                    scores['urgent'] += 7
                if self._media_is_payment_media(media_info) or self._is_voice_payment_request(media_info):
                    scores['payment'] += 7
                if self._is_voice_reminder(media_info):
                    scores['event'] += 6

        if self._contains_payment_intent(text):
            scores['payment'] += 7
        if self._contains_strong_urgent(text) and not self._has_negated_urgency(text):
            scores['urgent'] += 7
        if self._contains_event(text):
            scores['event'] += 5
        if self._contains_promotion(text):
            scores['promotion'] += 5
        if self._contains_greeting(text):
            scores['greeting'] += 5
        if self._contains_spam_signals(text):
            scores['spam'] += 4

        if any(token in text for token in ['lottery', 'prize', 'click here', 'otp sharing', 'suspicious link', 'suspicious links']):
            scores['spam'] += 4
        if any(token in text for token in ['otp', 'verify now', 'password reset', 'account blocked', 'wallet verification failed', 'reply with the 6 digit']):
            scores['scam'] += 5

        if conversation_type == 'business' and business is not None:
            if business_verified:
                if self._looks_like_business_update(text, business_category):
                    scores['business_update'] += 6
                if self._contains_event(text):
                    scores['event'] += 2
            if business_category in {'bank', 'insurance', 'finance', 'payment', 'wallet'}:
                scores['payment'] += 3
            if business_category in {'ecommerce', 'marketplace', 'travel', 'food_delivery', 'restaurant_dining', 'retail', 'fashion', 'beauty', 'hotel', 'streaming', 'grocery', 'grocery_delivery', 'quick_commerce', 'cinema', 'events'}:
                scores['promotion'] += 3
            if business_category in {'healthcare', 'healthcare_product', 'insurance', 'utilities', 'vehicle_service', 'traffic_challan', 'ecommerce_delivery', 'logistics', 'airline', 'telecom', 'security', 'bank'}:
                scores['event'] += 2
                scores['business_update'] += 2
            if self._looks_like_business_update(text, business_category):
                scores['business_update'] += 3

        if conversation_type == 'group':
            if self._is_group_forward(context, text):
                scores['forward'] += 8
            if self._is_group_scam(text, business, media_info):
                scores['scam'] += 8
            if self._is_group_greeting(text):
                scores['greeting'] += 6
            if self._is_group_urgent(text, group_type):
                scores['urgent'] += 6
            if self._is_group_event(text, group_type):
                scores['event'] += 6
            if self._is_group_promotion(text, group_type):
                scores['promotion'] += 6
            if self._is_group_personal(text, context):
                scores['personal'] += 4
            if group_muted and self._contains_spam_signals(text):
                scores['spam'] += 3
            if group_type in {'marketplace', 'local_food', 'real_estate'}:
                scores['promotion'] += 3
            if group_type in {'school_group', 'college_faculty', 'college_students', 'society'}:
                scores['event'] += 3

        if conversation_type == 'personal':
            if forwarded_count > 0 and self._contains_spam_signals(text):
                scores['spam'] += 6
            if self._contains_greeting(text):
                if self._is_greeting_only(text):
                    scores['greeting'] += 6
                else:
                    scores['unknown'] += 2
            if self._is_simple_question(text):
                scores['unknown'] += 3
            if self._contains_event(text) and not self._contains_promotion(text):
                scores['event'] += 3
            if self._contains_promotion(text):
                scores['promotion'] += 3

        if history_hint == 'forward':
            scores['forward'] += 4
        if history_hint == 'promotion':
            scores['promotion'] += 3
        if history_hint == 'scam':
            scores['scam'] += 4

        if trust < 0.25 and any(token in text for token in ['lottery', 'prize', 'click here', 'otp', 'verify', 'suspicious link']):
            scores['scam'] += 2
        if trust < 0.35 and any(token in text for token in ['offer', 'discount', 'cashback', 'coupon']):
            scores['promotion'] += 2

        best_type, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score <= 0:
            return 'unknown'
        return best_type

    def _is_group_forward(self, context: RoutingContext, text: str) -> bool:
        forwarded_count = _coerce_int(context.message.get('forwarded_count'), 0)
        if forwarded_count <= 0:
            return False
        if 'fwd as received' in text or 'forwarded as received' in text or 'fwd:' in text:
            return True
        if forwarded_count > 4 and any(token in text for token in ['forwarded', 'fwd', 'forwarding']):
            return True
        return False

    def _is_group_scam(self, text: str, business: Optional[pd.Series], media_info: Optional[MediaInformation]) -> bool:
        if self._is_scam(text, business, media_info):
            return True
        if media_info is not None and self._media_is_scam_media(media_info):
            return True
        scam_tokens = {
            'account blocked', 'verify now', 'security alert', 'login code', 'confirm password', 'password reset', 'otp', 'pin', 'account will be blocked', 'expire today', 'wallet verification failed', 'reply with the 6 digit', 'keep your account active', 'unauthorized'
        }
        return any(token in text for token in scam_tokens)

    def _is_group_urgent(self, text: str, group_type: str) -> bool:
        urgent_tokens = {
            'urgent', 'immediately', 'asap', 'alert', 'deadline', 'action required', 'blocked', 'must pay', 'critical', 'security alert', 'emergency', 'incident', 'eod', 'last-minute', 'last minute', 'heads-up', 'quick heads-up', 'please fill', 'fill now', 'need to'
        }
        if self._has_negated_urgency(text):
            return False
        if any(token in text for token in urgent_tokens) and not self._is_group_promotion(text, group_type):
            return True
        if group_type in {'safety', 'school_group', 'college_faculty'} and any(token in text for token in ['deadline', 'submit', 'pickup', 'service stops', 'blocked', 'immediately', 'fill']) and not any(token in text for token in ['today', 'tomorrow', 'schedule', 'route', 'notice', 'circular', 'review']):
            return True
        return False

    def _is_group_greeting(self, text: str) -> bool:
        greetings = ['good morning', 'good evening', 'good night', 'hello everyone', 'hi everyone', 'hey everyone', 'greetings', 'stay positive', 'hope today', 'dear all', 'good vibes', 'peaceful for all']
        if any(phrase in text for phrase in greetings) and not any(token in text for token in ['urgent', 'alert', 'action required', 'please respond', 'meeting', 'deadline', 'pickup', 'schedule', 'form', 'circular', 'consent', 'review', 'route', 'bus', 'notice']):
            return True
        if text.startswith('good morning') or text.startswith('good evening') or text.startswith('good night'):
            return True
        return False

    def _is_group_promotion(self, text: str, group_type: str) -> bool:
        promo_tokens = {
            'offer', 'sale', 'discount', 'deal', 'buy', 'order', 'shop', 'subscribe', 'unsubscribe', 'try', 'limited', 'free', 'review', 'price', 'cash', 'dm if interested', 'terms apply', 'reply stop', 'use it now', 't&c', 'tap below', 'get 50%', 'percent off', 'expires soon', 'selling', 'for sale', 'interested', 'available', 'listing', 'swap', 'wanted'
        }
        if any(token in text for token in promo_tokens):
            return True
        return group_type in {'marketplace', 'local_food', 'real_estate'}

    def _is_group_personal(self, text: str, context: RoutingContext) -> bool:
        personal_signals = ['@', 'can you', 'could you', 'call me', 'let me know', 'need you', 'urgent request', 'can we', 'when you get', 'just checking', 'join with']
        if any(token in text for token in personal_signals):
            return True
        if context.group_membership is not None and _coerce_int(context.group_membership.get('group_admin'), 0) == 1:
            return any(token in text for token in ['you', 'your', 'let me know', 'can you', '@'])
        return False

    def _is_group_event(self, text: str, group_type: str) -> bool:
        event_tokens = {
            'event', 'meeting', 'reminder', 'schedule', 'appointment', 'consent', 'approval', 'notice', 'update', 'circular', 'session', 'class', 'campus', 'faculty', 'club', 'program', 'show', 'sheet', 'form is open', 'rsvp', 'next sunday', 'tomorrow', 'today', 'route', 'timing', 'pickup', 'parents', 'bus', 'water pressure', 'valve', 'change', 'review', 'flat', 'supply'
        }
        if any(token in text for token in event_tokens) and not self._is_group_promotion(text, group_type):
            return True
        return group_type in {'school_group', 'college_faculty', 'college_students', 'society'}

    def _default_group_message_type(self, group_type: str) -> str:
        if group_type in {'marketplace', 'local_food', 'real_estate'}:
            return 'promotion'
        if group_type == 'safety':
            return 'urgent'
        if group_type == 'finance_help':
            return 'business_update'
        if group_type in {'school_group', 'college_faculty', 'college_students', 'society'}:
            return 'event'
        if group_type in {'family', 'extended_family', 'friends', 'alumni', 'book_club', 'dance_class', 'caregiving', 'sports'}:
            return 'personal'
        return 'personal'

    def _history_hint(self, history: HistoricalRetrieval) -> str | None:
        if history.related_messages.empty:
            return None
        combined = ' '.join(str(text).lower() for text in history.related_messages['message_text'].fillna('').astype(str))
        if 'fwd as received' in combined or 'forwarded as received' in combined or 'fwd:' in combined:
            return 'forward'
        if 'offer' in combined or 'discount' in combined or 'unsubscribe' in combined:
            return 'promotion'
        if 'otp' in combined or 'account blocked' in combined or 'verify now' in combined:
            return 'scam'
        return None

    def _is_greeting_only(self, text: str) -> bool:
        if not self._contains_greeting(text):
            return False
        if '?' in text:
            return False
        return len(text.split()) <= 10

    def _is_simple_question(self, text: str) -> bool:
        if not text or '?' not in text:
            return False
        if any(token in text for token in ['urgent', 'otp', 'payment', 'offer', 'discount', 'meeting', 'reminder', 'schedule']):
            return False
        return len(text.split()) <= 20

    def _has_negated_urgency(self, text: str) -> bool:
        negations = [
            'nothing urgent', 'not urgent', 'no need to respond', 'no pressure', 'nothing dramatic',
            "don't call now", "dont call now", 'dont call', 'do not call', 'no urgency', 'can wait'
        ]
        return any(phrase in text for phrase in negations)

    def _looks_like_business_update(self, text: str, business_category: str) -> bool:
        if not text:
            return False
        if any(token in text for token in ['thank you', 'feedback', 'review', 'order', 'delivery', 'packed', 'expected', 'reached', 'details', 'update', 'reminder', 'advisory', 'safety advisory']):
            return True
        if business_category in {'healthcare', 'healthcare_product', 'insurance', 'utilities', 'vehicle_service', 'traffic_challan', 'bank', 'security'} and any(token in text for token in ['appointment', 'prescription', 'pickup', 'claim', 'timing', 'consent', 'notice', 'circular']):
            return True
        return False

    def _contains_payment_intent(self, text: str) -> bool:
        payment_tokens = {
            'payment reminder', 'due payment', 'due amount', 'payment update', 'bill due', 'invoice', 'amount due', 'pay now', 'verify now', 'otp', 'transaction', 'payment request', 'pay by', 'expires today', 'payment link', 'upi', 'emi', 'recharge', 'bank', 'statement', 'account update', 'billing'
        }
        return any(token in text for token in payment_tokens)

    def _contains_strong_urgent(self, text: str) -> bool:
        urgent_tokens = {
            'urgent', 'immediately', 'asap', 'alert', 'deadline', 'action required', 'critical', 'emergency', 'incident', 'must pay', 'blocked', 'security alert', 'eod', 'last-minute', 'last minute', 'heads-up', 'quick heads-up', 'please fill', 'fill now', 'need to', 'help', 'hospital', 'immediate'
        }
        return any(token in text for token in urgent_tokens)

    def _contains_promotion(self, text: str) -> bool:
        promo_tokens = {
            'offer', 'sale', 'discount', 'deal', 'buy', 'shop', 'subscribe', 'unsubscribe', 'try', 'limited', 'free', 'review', 'price', 'cash', 'cashback', 'coupon', 'festival sale', 'tap below', 'reply stop', 't&c', 'get 50%', 'percent off', 'expires soon', 'discount code', 'selling', 'for sale', 'interested', 'dm if interested', 'available', 'listing', 'swap', 'wanted'
        }
        return any(token in text for token in promo_tokens)

    def _contains_event(self, text: str) -> bool:
        event_tokens = {
            'event', 'meeting', 'seminar', 'webinar', 'celebration', 'birthday', 'invitation', 'reminder', 'schedule', 'appointment', 'consent', 'approval', 'notice', 'update', 'circular', 'session', 'class', 'campus', 'faculty', 'club', 'program', 'show', 'form is open', 'rsvp', 'tomorrow', 'today', 'next sunday', 'tonight', 'pickup', 'registration', 'timing', 'school function'
        }
        return any(token in text for token in event_tokens)

    def _contains_spam_signals(self, text: str) -> bool:
        spam_tokens = {'forwarded', 'fwd', 'forwarding', 'unsubscribe', 'reply stop', 'stop', 'free', 'win', 'cash', 'discount', 'offer', 'deal'}
        return any(token in text for token in spam_tokens)

    def _contains_greeting(self, text: str) -> bool:
        return any(phrase in text for phrase in ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good night', 'dear all', 'dear customer', 'greetings', 'happy birthday', 'congratulations', 'thank you'])

    def _is_scam(self, text: str, business: Optional[pd.Series], media_info: Optional[MediaInformation] = None) -> bool:
        scam_tokens = {
            'scam', 'phishing', 'billing issue', 'password reset', 'verify now', 'account blocked', 'bank details', 'security alert', 'unauthorized', 'otp', 'pin', 'confirm password', 'login code', 'account will be blocked', 'wallet verification failed', 'reply with the 6 digit', 'keep your account active', 'otp sharing', 'suspicious link', 'suspicious links', 'lottery', 'prize', 'click here'
        }
        if any(token in text for token in scam_tokens):
            return True
        if business is not None and int(business.get('verified') or 0) == 0 and any(token in text for token in ['password', 'verify', 'login', 'transfer', 'otp', 'pin', 'account', 'payment']):
            return True
        if media_info is not None and self._media_is_scam_media(media_info):
            return True
        return False

    def _decide_reason(self, context: RoutingContext, history: HistoricalRetrieval, media_info: Optional[MediaInformation], action: str, message_type: str) -> str:
        message = context.message
        business = context.business_account
        forwarded_count = int(message.get('forwarded_count') or 0)
        text = str(message.get('message_text') or '').strip()
        trust = self._sender_trust_score(context)
        history_score = self._historical_behavior_score(context, history)
        media_category = self._media_analysis_field(media_info, 'category')
        media_urgency = self._media_analysis_field(media_info, 'urgency')
        media_confidence = self._media_analysis_confidence(media_info)

        if action == 'notify':
            if media_info is not None and media_info.analysis is not None:
                return f'The media looks like {media_category} with {media_urgency or "normal"} urgency, so it was routed immediately.'
            if message_type == 'urgent':
                return 'The message looks time-sensitive, so it was routed for immediate attention.'
            if message_type == 'payment':
                return 'The message appears payment-related, so it was prioritized for immediate attention.'
            if business is not None and int(business.get('verified') or 0) == 1 and trust >= 0.7:
                return 'A verified business message with relevant context was routed as a notification.'
            return 'The message had clear relevance or urgency, so it was routed as a notification.'

        if action == 'mute':
            if self._is_scam(text, business, media_info):
                return 'Suspicious or unsafe content was muted to reduce risk.'
            if forwarded_count > 2 and history_score <= 0:
                return 'Repeated or low-trust content was muted because it looked low-value.'
            return 'Low-value or low-trust content was muted.'

        if action == 'digest':
            if media_info is not None and media_info.analysis is not None:
                return f'The media looked like {media_category} content, so it was batched into the digest.'
            if message_type == 'promotion':
                return 'Promotional content with low urgency was batched for later review.'
            if message_type == 'event':
                return 'Routine informational content was batched into the digest.'
            return 'Low-priority content was grouped into the digest.'

        return 'The routing decision was based on relevance, risk, and user context.'

    def _decide_confidence(self, context: RoutingContext, history: HistoricalRetrieval, media_info: Optional[MediaInformation], action: str, message_type: str) -> float:
        trust = self._sender_trust_score(context)
        history_score = self._historical_behavior_score(context, history)
        business = context.business_account
        business_verified = int(business.get('verified') or 0) == 1 if business is not None else False
        forwarded_count = int(context.message.get('forwarded_count') or 0)
        text = str(context.message.get('message_text') or '').lower()
        conversation_type = str(context.message.get('conversation_type') or '').lower()
        scam = self._is_scam(text, business, media_info)
        urgency = self._is_urgent(text, conversation_type)
        evidence_count = len(history.evidence_message_ids)
        notification_summary = context.notification_summary
        user = context.user
        sender_user = context.sender_user
        media_confidence = self._media_analysis_confidence(media_info)

        score = 0.2

        if business_verified:
            score += 0.18
        if business is not None:
            official_domain = str(business.get('official_domain') or '').lower()
            sender_domain = str(business.get('domain_used_by_sender') or '').lower()
            if official_domain and sender_domain and official_domain == sender_domain:
                score += 0.12
            reports = int(business.get('user_reports_30d') or 0)
            if reports >= 5:
                score -= 0.14
            elif reports >= 2:
                score -= 0.08

        if trust >= 0.8:
            score += 0.14
        elif trust >= 0.6:
            score += 0.08
        elif trust <= 0.3:
            score -= 0.1

        if history_score > 0.2:
            score += 0.12
        elif history_score > 0.0:
            score += 0.06
        elif history_score < -0.2:
            score -= 0.08
        else:
            score -= 0.04

        if evidence_count >= 3:
            score += 0.08
        elif evidence_count >= 1:
            score += 0.04
        else:
            score -= 0.05

        if user is not None:
            user_replies = int(user.get('messages_replied_30d') or 0)
            user_opens = int(user.get('messages_opened_30d') or 0)
            if user_replies > 3 or user_opens > 6:
                score += 0.08
            elif user_replies > 0 or user_opens > 0:
                score += 0.04

        if notification_summary.shape[0] > 0:
            total_sent = int(notification_summary['notifications_sent'].sum() or 0)
            dismissed = int(notification_summary['notifications_dismissed'].sum() or 0)
            if total_sent > 0:
                dismissal_rate = dismissed / total_sent
                if dismissal_rate > 0.6:
                    score -= 0.08
                elif dismissal_rate > 0.3:
                    score -= 0.04

        if self._contains_payment_intent(text):
            score += 0.08
        if self._is_otp(text):
            score += 0.10
        if self._contains_strong_urgent(text) and not self._has_negated_urgency(text):
            score += 0.1
        if urgency:
            score += 0.04
        if self._contains_promotion(text):
            score += 0.03
        if self._contains_event(text):
            score += 0.03
        if self._contains_greeting(text) and self._is_greeting_only(text):
            score += 0.02

        if sender_user is None:
            score -= 0.08
        elif bool(sender_user.get('verified') or 0):
            score += 0.04

        if forwarded_count >= 5:
            score -= 0.12
        elif forwarded_count >= 3:
            score -= 0.08
        elif forwarded_count >= 1:
            score -= 0.04

        if scam or self._contains_spam_signals(text):
            score -= 0.16

        conflict_signals = 0
        if self._contains_payment_intent(text) and self._contains_promotion(text):
            conflict_signals += 1
        if self._contains_event(text) and self._contains_strong_urgent(text):
            conflict_signals += 1
        if self._contains_greeting(text) and self._contains_payment_intent(text):
            conflict_signals += 1
        if self._contains_promotion(text) and self._contains_event(text):
            conflict_signals += 1
        if conflict_signals > 0:
            score -= min(conflict_signals * 0.04, 0.12)

        if action == 'notify':
            score += 0.06 if message_type in {'urgent', 'payment', 'scam'} else 0.03
        elif action == 'mute':
            score -= 0.08
        elif action == 'digest':
            score += 0.02 if message_type in {'promotion', 'event', 'greeting'} else 0.0

        if media_confidence > 0:
            score += min((media_confidence - 0.5) * 0.2, 0.12)

        return round(min(max(score, 0.0), 1.0), 2)

    def _sender_trust_score(self, context: RoutingContext) -> float:
        business = context.business_account
        score = 0.5
        if business is None:
            return score

        verified = int(business.get('verified') or 0) == 1
        official_domain = str(business.get('official_domain') or '').lower()
        sender_domain = str(business.get('domain_used_by_sender') or '').lower()
        reports = int(business.get('user_reports_30d') or 0)
        account_age = int(business.get('account_age_days') or 0)

        if verified:
            score += 0.25
        if official_domain and sender_domain and official_domain == sender_domain:
            score += 0.15
        if reports >= 5:
            score -= 0.2
        elif reports >= 2:
            score -= 0.1
        if account_age > 365:
            score += 0.1
        if account_age > 1095:
            score += 0.05
        if not verified and sender_domain and official_domain and sender_domain != official_domain:
            score -= 0.1

        return round(min(max(score, 0.0), 1.0), 2)

    def _historical_behavior_score(self, context: RoutingContext, history: HistoricalRetrieval) -> float:
        if history.related_events.empty and context.user_business_history.empty:
            return 0.0

        events = history.related_events
        score = 0.0

        if not events.empty:
            opened_rate = float(events['message_opened'].mean() or 0.0)
            replied_rate = float(events['message_replied'].mean() or 0.0)
            dismissed_rate = float(events['notification_dismissed'].mean() or 0.0)
            reported_rate = float(events['message_reported'].mean() or 0.0)
            muted_rate = float(events['muted_after_message'].mean() or 0.0)

            score += (opened_rate - 0.6) * 0.2
            score += (replied_rate - 0.2) * 0.25
            score -= dismissed_rate * 0.3
            score -= reported_rate * 0.4
            score -= muted_rate * 0.35

        business_history = context.user_business_history
        if not business_history.empty:
            total_opened = int(business_history['messages_opened_30d'].sum() or 0)
            total_dismissed = int(business_history['messages_dismissed_30d'].sum() or 0)
            total_replied = int(business_history['messages_replied_30d'].sum() or 0)
            activity_count = int(business_history['activity_count_180d'].sum() or 0)
            last_active = business_history['last_activity_at'].max()

            if total_opened > total_dismissed:
                score += 0.1
            if total_replied > 0:
                score += 0.1
            if activity_count > 2:
                score += 0.05
            if pd.notna(last_active) and last_active >= pd.Timestamp.now() - pd.Timedelta(days=30):
                score += 0.05
            if total_dismissed > total_opened:
                score -= 0.1

        return round(min(max(score, -1.0), 1.0), 2)

    def _is_urgent(self, text: str, conversation_type: str | None) -> bool:
        urgent_tokens = {
            'otp', 'payment reminder', 'due payment', 'immediately', 'now', 'emergency', 'alert', 'incident', 'urgent', 'calendar', 'meeting', 'deadline', 'action required', 'heads-up', 'quick heads-up', 'please fill', 'fill now', 'need to'
        }
        if self._has_negated_urgency(text):
            return False
        return any(token in text for token in urgent_tokens)

    def _is_otp(self, text: str) -> bool:
        return 'otp' in text or 'one time password' in text or 'verification code' in text

    def _media_analysis_field(self, media_info: Optional[MediaInformation], field: str) -> str:
        if media_info is None or media_info.analysis is None:
            return ''
        return str(media_info.analysis.get(field, '') or '').lower()

    def _media_analysis_confidence(self, media_info: Optional[MediaInformation]) -> float:
        if media_info is None or media_info.analysis is None:
            return 0.0
        try:
            return float(media_info.analysis.get('confidence', 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _media_is_scam_media(self, media_info: Optional[MediaInformation]) -> bool:
        summary = self._media_analysis_field(media_info, 'summary')
        category = self._media_analysis_field(media_info, 'category')
        return 'scam' in summary or 'scam' in category or 'fraud' in summary

    def _media_is_payment_media(self, media_info: Optional[MediaInformation]) -> bool:
        summary = self._media_analysis_field(media_info, 'summary')
        category = self._media_analysis_field(media_info, 'category')
        return 'payment' in summary or 'payment' in category or 'bill' in summary or 'due' in summary

    def _media_is_emergency_media(self, media_info: Optional[MediaInformation]) -> bool:
        urgency = self._media_analysis_field(media_info, 'urgency')
        summary = self._media_analysis_field(media_info, 'summary')
        return 'emergency' in summary or urgency in {'high', 'urgent', 'critical'}

    def _media_is_invitation_media(self, media_info: Optional[MediaInformation]) -> bool:
        summary = self._media_analysis_field(media_info, 'summary')
        category = self._media_analysis_field(media_info, 'category')
        return 'invitation' in summary or 'invitation' in category or 'event' in category

    def _media_is_delivery_media(self, media_info: Optional[MediaInformation]) -> bool:
        summary = self._media_analysis_field(media_info, 'summary')
        category = self._media_analysis_field(media_info, 'category')
        return 'delivery' in summary or 'shipment' in category or 'tracking' in summary

    def _media_is_advertisement_media(self, media_info: Optional[MediaInformation]) -> bool:
        summary = self._media_analysis_field(media_info, 'summary')
        category = self._media_analysis_field(media_info, 'category')
        return 'advertisement' in summary or 'advertisement' in category or 'promo' in summary or 'promotion' in category

    def _media_is_school_media(self, media_info: Optional[MediaInformation]) -> bool:
        summary = self._media_analysis_field(media_info, 'summary')
        category = self._media_analysis_field(media_info, 'category')
        return 'school' in summary or 'school' in category or 'notice' in summary and 'school' in summary

    def _is_voice_payment_request(self, media_info: Optional[MediaInformation]) -> bool:
        if media_info is None or media_info.analysis is None:
            return False
        payment_request = self._media_analysis_field(media_info, 'payment_request')
        return payment_request in {'yes', 'true', 'payment', 'request', 'requested'}

    def _is_voice_reminder(self, media_info: Optional[MediaInformation]) -> bool:
        if media_info is None or media_info.analysis is None:
            return False
        reminder = self._media_analysis_field(media_info, 'reminder')
        return reminder in {'payment reminder', 'reminder', 'due date', 'bill due', 'follow up'}
