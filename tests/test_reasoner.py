import unittest

import pandas as pd

from code.agents.reasoner import RuleBasedReasoner
from code.models.schemas import HistoricalRetrieval, RoutingContext


class ReasonerNAHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reasoner = RuleBasedReasoner()

    def test_group_personal_detection_handles_pandas_na(self) -> None:
        message = pd.Series(
            {
                'message_id': 'm1',
                'message_text': 'Can you help me with this?',
                'conversation_type': 'group',
                'forwarded_count': 0,
                'user_id': 'u1',
                'group_id': 'g1',
                'business_id': pd.NA,
            }
        )
        group_membership = pd.Series({'group_id': 'g1', 'user_id': 'u1', 'group_admin': pd.NA, 'group_muted_by_user': pd.NA})
        context = RoutingContext(
            message_id='m1',
            message=message,
            user=None,
            sender_user=None,
            group=pd.Series({'group_id': 'g1', 'group_type': 'family'}),
            group_membership=group_membership,
            group_members=pd.DataFrame(columns=['group_id', 'user_id', 'group_admin', 'group_muted_by_user']),
            business_account=None,
            user_business_history=pd.DataFrame(columns=['user_id', 'business_id']),
            notification_summary=pd.DataFrame(columns=['user_id', 'date', 'notifications_sent', 'notifications_dismissed']),
        )
        history = HistoricalRetrieval(
            message_id='m1',
            evidence_message_ids=[],
            related_messages=pd.DataFrame(columns=['message_text']),
            related_events=pd.DataFrame(columns=['event_type']),
        )

        self.assertTrue(self.reasoner._is_group_personal('can you help me with this?', context))
        self.assertIn(
            self.reasoner._decide_message_type(context, history, None),
            ['personal', 'urgent', 'event', 'payment', 'business_update', 'promotion', 'greeting', 'forward', 'spam', 'scam', 'unknown'],
        )


if __name__ == '__main__':
    unittest.main()
