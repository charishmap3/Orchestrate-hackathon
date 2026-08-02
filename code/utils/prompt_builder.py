from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_gemini_prompt(
    message: dict,
    user: dict,
    group: Optional[dict],
    business: Optional[dict],
    notification_summary: list[dict],
    historical_messages: list[dict],
    historical_events: list[dict],
    evidence: list[str],
    media_info: dict,
) -> str:
    lines: list[str] = [
        'You are an intelligent WhatsApp Notification Router. Use the supplied context to choose the single best routing decision.',
        'Respond only with valid JSON and do not include markdown, explanations, or any extra text.',
        '',
        'Valid output format:',
        '{',
        '  "action":"notify",',
        '  "message_type":"payment",',
        '  "reason":"Brief explanation.",',
        '  "confidence":0.95,',
        '  "media_analysis": {',
        '    "summary":"",',
        '    "importance":"",',
        '    "risk":"",',
        '    "confidence":0.92',
        '  }',
        '}',
        '',
        'Valid actions: notify, digest, mute.',
        'Do not use message IDs in reasoning.',
        'Use only the supplied context.',
        '',
        'Current Message:',
        f'- text: "{message.get("message_text", "")}"',
        f'- forwarded_count: {message.get("forwarded_count", 0)}',
        f'- conversation_type: {message.get("conversation_type", "")}',
        f'- media_type: {media_info.get("media_type") or "none"}',
    ]

    if media_info.get('media_summary'):
        lines.append(f'- media_summary: {media_info["media_summary"]}')

    if media_info.get('media_analysis'):
        lines.append('')
        lines.append('Media Analysis:')
        for key, value in media_info['media_analysis'].items():
            if value is not None and value != '':
                lines.append(f'- {key}: {value}')

    lines.extend([
        '',
        'User Profile:',
        f'- user_id: {user.get("user_id", "")}',
        f'- messages_replied_30d: {user.get("messages_replied_30d", 0)}',
        f'- notifications_dismissed: {notification_summary[0].get("notifications_dismissed", 0) if notification_summary else 0}',
    ])

    if group is not None:
        lines.extend([
            '',
            'Group Information:',
            f'- group_id: {group.get("group_id", "")}',
            f'- group_type: {group.get("group_type", "")}',
        ])

    if business is not None:
        lines.extend([
            '',
            'Business Information:',
            f'- business_id: {business.get("business_id", "")}',
            f'- verified: {business.get("verified", 0)}',
            f'- category: {business.get("category", "")}',
        ])

    lines.extend([
        '',
        'Historical Messages:',
    ])
    if historical_messages:
        for item in historical_messages[:3]:
            lines.append(f'- {item.get("created_at", "")}: {item.get("message_text", "")[:120]}')
    else:
        lines.append('- none')

    lines.extend([
        '',
        'Historical Events:',
    ])
    if historical_events:
        for item in historical_events[:3]:
            lines.append(f'- {item.get("event_type", "")}: {item.get("description", "")[:120]}')
    else:
        lines.append('- none')

    lines.extend([
        '',
        'Notification Summary:',
    ])
    if notification_summary:
        for item in notification_summary[:3]:
            lines.append(f'- {item.get("date", "")} dismissed={item.get("notifications_dismissed", 0)}')
    else:
        lines.append('- none')

    lines.extend([
        '',
        'Retrieved Evidence message IDs:',
        ', '.join(evidence) if evidence else 'none',
        '',
        'Instructions:',
        '1. Consider urgency, business verification, forwarded count, user notification behaviour, historical interaction, muted groups, repeated promotions, scams, safety, spam, user-business relationship, and media content.',
        '2. If the message is a verified banking/payment notification, known scam, muted group, or repeated spam, prefer a high-confidence rule-based decision.',
        '3. Otherwise, choose the best action using all context.',
        '4. Return confidence as a float between 0.0 and 1.0.',
    ])

    return '\n'.join(lines)
