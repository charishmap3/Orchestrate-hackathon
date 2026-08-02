# Orchestrate: Intelligent Message Routing for High-Noise Messaging Streams

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Verified-success)

## Overview

Orchestrate is a hybrid rule-and-AI routing system designed for the HackerRank Orchestrate challenge. The project processes messaging data and decides whether each incoming message should be treated as a high-priority notification, a lower-priority digest, or a muted item.

The system is built to handle the real-world noise of modern messaging environments, including direct messages, business conversations, group chats, promotions, scams, image posts, and voice notes. Instead of treating every message equally, it triages content using context, behavior history, and multimodal signals.

## Key Features

- Intelligent message routing across three actions: Notify, Digest, and Mute
- Rule-based reasoning for fast, deterministic decisions
- Gemini AI fallback for ambiguous or multimodal cases
- Context-aware routing using message text, conversation type, and user metadata
- Historical message retrieval for evidence-based classification
- Business verification-aware routing
- User behavior analysis based on engagement and notification history
- Image support for multimodal routing
- Voice note support for multimodal routing
- Confidence scoring for each routing decision
- Output generation to CSV in the required schema
- Evaluation metrics for action accuracy, message-type accuracy, precision, recall, F1, and confusion matrix

## Architecture

```text
Dataset
↓
Dataset Loader
↓
Context Builder
↓
Historical Retrieval
↓
Media Processor
↓
Rule-Based Reasoner
↓
Gemini AI Fallback
↓
Routing Decision
↓
Output Writer
↓
Evaluation
```

## Project Structure

```text
.
├── AGENTS.md                    # Guidance for coding agents and transcript logging
├── CLAUDE.md                    # Additional local workflow notes
├── code/
│   ├── agents/                  # Routing, retrieval, reasoning, and media logic
│   ├── evaluation/              # Metrics and evaluation utilities
│   ├── models/                  # Dataclasses and shared schemas
│   ├── prompts/                 # Prompt assets used by the reasoning flow
│   ├── services/                # External service integrations
│   ├── utils/                   # Helper utilities
│   └── main.py                  # End-to-end pipeline entry point
├── dataset/
│   ├── messages.csv             # Incoming messages to route
│   ├── sample_messages.csv      # Labeled sample messages used for evaluation
│   ├── users.csv                # User metadata and engagement signals
│   ├── groups.csv               # Group metadata
│   ├── group_members.csv        # Group membership and mute/admin flags
│   ├── business_accounts.csv    # Business sender metadata and verification status
│   ├── user_business_history.csv
│   ├── message_history.csv      # Historical messages used as evidence
│   ├── message_events.csv       # Message interaction events
│   ├── images.csv               # Image metadata and file paths
│   ├── voice_notes.csv          # Voice note metadata and file paths
│   ├── daily_notification_summary.csv
│   └── media/                   # Media assets used by the pipeline
├── tests/                       # Regression tests for reasoning behavior
└── README.md                    # Project documentation
```

## Tech Stack

- Python
- Pandas
- Google Gemini API
- CSV-based dataset processing
- Logging and structured runtime reporting
- Rule-based AI reasoning
- Hybrid AI routing

## Workflow

A message moves through the pipeline in the following order:

1. The dataset loader reads the CSV files from the dataset directory.
2. The context builder assembles the relevant user, group, business, and message metadata.
3. Historical evidence is retrieved from prior messages and related events.
4. The media processor resolves image or voice note paths when present.
5. The rule-based reasoner produces an initial routing decision.
6. The routing engine may invoke Gemini for low-confidence, ambiguous, or multimodal cases.
7. The output writer normalizes and saves the final action, message type, reason, confidence, and evidence IDs to CSV.
8. The evaluator compares predictions against the sample labels and prints accuracy and confusion metrics.

## Routing Logic

The routing engine can produce three actions:

| Action | Purpose |
|---|---|
| Notify | Use when the message appears urgent, transactional, personal, or otherwise important |
| Digest | Use when the message is informational or should be batched for later review |
| Mute | Use for spam, scam-like, or clearly unwanted messages |

The decision is shaped by the following signals:

- Business verification: verified businesses receive different treatment than unverified ones.
- Forward count: repeated or forwarded content can indicate spam-like behavior or promotion cascades.
- Historical evidence: prior messages and interactions provide context for repetition and intent.
- User engagement: user behavior and notification history influence how aggressively a message should be surfaced.
- DND-style signals: muted group conditions and other notification settings affect the routing decision.
- Group metadata: group type and membership flags influence whether a message should be treated as personal, promotional, or urgent.
- Media analysis: image and voice content can add urgency, payment, scam, or event cues.
- Confidence score: the rule engine assigns a confidence score, and Gemini is engaged when confidence is low or the case is ambiguous.
- Gemini fallback: if Gemini is unavailable or quota-limited, the system automatically returns to the rule engine.

## Hybrid AI Design

The system uses a hybrid design:

- The Rule Engine handles the majority of messages and provides a strong baseline.
- Gemini is used selectively for:
  - low-confidence decisions
  - image-based cases
  - voice-based cases
  - ambiguous or conflicting signals

If Gemini is unavailable, the pipeline continues without interruption and falls back to the rule-based reasoner.

## Evaluation

Evaluation is performed after the main run. The evaluator loads the generated output and compares it with the labeled sample messages.

The project reports:

- Action Accuracy
- Message Type Accuracy
- Overall Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics are printed at the end of the run and help assess how well the routing logic performs on the provided sample set.

## Output

The pipeline writes a CSV file named `dataset/output.csv` with these fields:

| Column | Description |
|---|---|
| `message_id` | Identifier for the message |
| `action` | One of `notify`, `digest`, or `mute` |
| `message_type` | The predicted message category |
| `reason` | Short human-readable rationale |
| `confidence` | A numeric score from `0.0` to `1.0` |
| `evidence_message_ids` | Related historical message IDs or `none` |

## Installation

```bash
git clone https://github.com/your-org/hackerrank-orchestrate-august26-main.git
cd hackerrank-orchestrate-august26-main

python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r code/requirements.txt
```

Create a `.env` file in the project root and add your Gemini key if you want to enable the AI fallback path:

```env
GEMINI_API_KEY=your_key_here
```

Run the full pipeline:

```bash
python code/main.py
```

The script writes the results to `dataset/output.csv` and prints a summary of routing behavior and evaluation metrics.

## Sample Console Output

The following is an example of the kind of output produced by the verified run:

```text
==================================================
ORCHESTRATE EXECUTION SUMMARY
==================================================
Messages Processed : 110
Notify             : 63
Digest             : 17
Mute               : 30

Images Processed   : 15
Voice Notes        : 8

Rule Decisions     : 109
Gemini Decisions   : 1

Action Accuracy    : 0.667
Message Accuracy   : 0.567

Output File        : .../dataset/output.csv
==================================================
```

## Future Improvements

Potential next steps for the project include:

- Better multimodal understanding for images and voice content
- LLM fine-tuning for domain-specific routing behavior
- Retrieval-augmented generation for richer historical reasoning
- Learning from user feedback to improve decision quality over time
- Multi-agent orchestration for specialized routing tasks
- Real-time deployment for live messaging environments

## Screenshots

Placeholder screenshots for the project showcase:

- Architecture pipeline diagram
- Sample routing output from `dataset/output.csv`
- Evaluation metrics summary
- Example of a multimodal message being routed

## License

This project is licensed under the MIT License.

## Contributors

Add contributor names here:

- [Your Name]
- [Your Team]
