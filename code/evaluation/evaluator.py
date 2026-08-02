from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from code.agents.router import RoutingEngine
from code.agents.retrieval import retrieve_historical_evidence_for_row
from code.agents.context_builder import build_routing_context_for_row
from code.evaluation.metrics import EvaluationMetrics, build_confusion_matrix, classification_metrics
from code.models.schemas import DataCatalog


LOGGER = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER

    def evaluate(self, predictions_path: Path, sample_path: Path, catalog: Optional[DataCatalog] = None, router: Optional[RoutingEngine] = None) -> Optional[EvaluationMetrics]:
        self.logger.info('Evaluating predictions against sample labels')
        if not sample_path.exists():
            self.logger.warning('Sample file does not exist.')
            return None

        sample = pd.read_csv(sample_path)
        if sample.empty:
            self.logger.warning('Sample file is empty.')
            return None

        predictions = pd.DataFrame()
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)

        if not predictions.empty:
            merged = sample[['message_id', 'action', 'message_type']].merge(
                predictions[['message_id', 'action', 'message_type']],
                on='message_id',
                how='inner',
                suffixes=('_true', '_pred'),
            )
        else:
            merged = pd.DataFrame()

        if merged.empty and catalog is not None and router is not None:
            self.logger.info('No overlap with output.csv; predicting sample messages directly.')
            predictions = self._predict_sample_messages(sample, catalog, router)
            merged = sample[['message_id', 'action', 'message_type']].merge(
                predictions[['message_id', 'action', 'message_type']],
                on='message_id',
                how='inner',
                suffixes=('_true', '_pred'),
            )

        if merged.empty:
            self.logger.warning('No overlapping sample messages to evaluate.')
            return None

        action_accuracy = float((merged['action_true'] == merged['action_pred']).mean())
        message_type_accuracy = float((merged['message_type_true'] == merged['message_type_pred']).mean())
        overall_accuracy = float(((merged['action_true'] == merged['action_pred']) & (merged['message_type_true'] == merged['message_type_pred'])).mean())

        precision, recall, f1 = classification_metrics(merged['action_true'], merged['action_pred'])
        confusion_matrix = build_confusion_matrix(merged['action_true'], merged['action_pred'])

        return EvaluationMetrics(
            action_accuracy=action_accuracy,
            message_type_accuracy=message_type_accuracy,
            overall_accuracy=overall_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion_matrix,
        )

    def _predict_sample_messages(self, sample: pd.DataFrame, catalog: DataCatalog, router: RoutingEngine) -> pd.DataFrame:
        self.logger.info('Predicting sample messages using routing pipeline')
        rows: list[dict[str, str]] = []
        for _, sample_row in sample.iterrows():
            message_series = sample_row.loc[
                ['message_id', 'user_id', 'conversation_type', 'group_id', 'business_id', 'sender_user_id', 'created_at', 'message_text', 'media_type', 'media_id', 'forwarded_count']
            ]
            message_id = str(message_series['message_id'])
            try:
                context = build_routing_context_for_row(message_series, catalog)
                history = retrieve_historical_evidence_for_row(message_id, message_series, catalog)
                decision = router.route(context, history, message_row=message_series)
                rows.append({
                    'message_id': decision.message_id,
                    'action': decision.action,
                    'message_type': decision.message_type,
                })
            except Exception:
                self.logger.exception('Failed to predict sample message %s', message_id)
                rows.append({
                    'message_id': message_id,
                    'action': 'mute',
                    'message_type': 'unknown',
                })

        return pd.DataFrame(rows)

    def print_metrics(self, metrics: EvaluationMetrics) -> None:
        self.logger.info('Evaluation results:')
        print(f'Action Accuracy: {metrics.action_accuracy:.3f}')
        print(f'Message Type Accuracy: {metrics.message_type_accuracy:.3f}')
        print(f'Overall Accuracy: {metrics.overall_accuracy:.3f}\n')

        print('Precision:')
        for label, value in sorted(metrics.precision.items()):
            print(f'  {label}: {value:.3f}')

        print('\nRecall:')
        for label, value in sorted(metrics.recall.items()):
            print(f'  {label}: {value:.3f}')

        print('\nF1 Score:')
        for label, value in sorted(metrics.f1_score.items()):
            print(f'  {label}: {value:.3f}')

        print('\nConfusion Matrix:')
        print(metrics.confusion_matrix.to_string())
