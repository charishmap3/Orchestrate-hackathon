from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.agents.loader import DatasetLoader
from code.agents.output_writer import OutputWriter
from code.agents.router import RoutingEngine
from code.evaluation.evaluator import Evaluator


LOGGER = logging.getLogger('orcchestrate')
LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)


def print_summary(output_df, writer, metrics, output_path: Path) -> None:
    action_counts = output_df['action'].value_counts().to_dict()
    notify_count = action_counts.get('notify', 0)
    digest_count = action_counts.get('digest', 0)
    mute_count = action_counts.get('mute', 0)

    print('==================================================')
    print('ORCHESTRATE EXECUTION SUMMARY')
    print('==================================================')
    print(f'Messages Processed : {len(output_df)}')
    print(f'Notify             : {notify_count}')
    print(f'Digest             : {digest_count}')
    print(f'Mute               : {mute_count}')
    print()
    print(f'Images Processed   : {writer.router.images_processed}')
    print(f'Voice Notes        : {writer.router.voice_notes_processed}')
    print()
    print(f'Rule Decisions     : {writer.router.rule_calls}')
    print(f'Gemini Decisions   : {writer.router.gemini_calls}')
    print()
    if metrics is not None:
        print(f'Action Accuracy    : {metrics.action_accuracy:.3f}')
        print(f'Message Accuracy   : {metrics.message_type_accuracy:.3f}')
    else:
        print('Action Accuracy    : n/a')
        print('Message Accuracy   : n/a')
    print()
    print(f'Output File        : {output_path}')
    print('==================================================')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Orchestrate full dataset processing and evaluation')
    parser.add_argument('--dataset-path', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        masked_key = api_key[:6] + '...' if len(api_key) > 6 else '******'
        LOGGER.info('GEMINI_API_KEY loaded from environment, starts with %s', masked_key)
    else:
        LOGGER.warning('GEMINI_API_KEY not found in environment after loading .env')

    loader = DatasetLoader(args.dataset_path)
    catalog = loader.load_all()

    output_path = args.dataset_path / 'output.csv'
    writer = OutputWriter(catalog)
    output_df = writer.write(output_path)

    evaluator = Evaluator()
    router = RoutingEngine(catalog)
    metrics = evaluator.evaluate(output_path, args.dataset_path / 'sample_messages.csv', catalog=catalog, router=router)

    print_summary(output_df, writer, metrics, output_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
