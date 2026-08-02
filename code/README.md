# Orchestrate: Intelligent Message Routing for High-Noise Messaging Streams

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![AI](https://img.shields.io/badge/AI-Gemini-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

# Overview

Orchestrate is a Hybrid AI Message Routing System developed for the HackerRank Orchestrate Hackathon.

The system intelligently classifies incoming messages into:

- 🔔 Notify
- 📰 Digest
- 🔕 Mute

Instead of treating every notification equally, the system analyzes message content, sender trust, historical interactions, user preferences, business verification, group metadata, and multimodal inputs to determine the most appropriate action.

The architecture combines a fast Rule-Based Reasoner with Gemini AI fallback, allowing accurate routing while minimizing unnecessary LLM calls.

---

# Features

- Hybrid Rule Engine + Gemini AI Routing
- Intelligent Notification Prioritization
- Business Verification Detection
- Sender Trust Scoring
- Historical Message Retrieval
- User Engagement Analysis
- Notification Dismissal History
- Group Metadata Analysis
- Forwarded Message Detection
- Spam & Scam Detection
- Dynamic Confidence Scoring
- Image Message Support
- Voice Note Support
- Automatic Rule-Based Fallback when Gemini quota is exceeded
- CSV Output Generation
- Automatic Evaluation Metrics

---

# System Architecture

```
Dataset
      │
      ▼
Dataset Loader
      │
      ▼
Context Builder
      │
      ▼
Historical Retrieval
      │
      ▼
Media Processor
      │
      ▼
Rule-Based Reasoner
      │
      ▼
Gemini AI (Low Confidence / Image / Voice)
      │
      ▼
Routing Decision
      │
      ▼
Output Writer
      │
      ▼
Evaluation
```

---

# Project Structure

```
Orchestrate-hackathon
│
├── code
│   ├── agents
│   ├── evaluation
│   ├── models
│   ├── prompts
│   ├── services
│   ├── utils
│   ├── requirements.txt
│   └── main.py
│
├── dataset
│   ├── messages.csv
│   ├── message_history.csv
│   ├── message_events.csv
│   ├── users.csv
│   ├── groups.csv
│   ├── group_members.csv
│   ├── business_accounts.csv
│   ├── user_business_history.csv
│   ├── images.csv
│   ├── voice_notes.csv
│   ├── output.csv
│   └── media
│
├── tests
│   └── test_reasoner.py
│
├── README.md
├── AGENTS.md
├── CLAUDE.md
└── .gitignore
```

---

# Routing Workflow

1. Load all datasets.
2. Build routing context.
3. Retrieve historical evidence.
4. Process image or voice media (if available).
5. Apply Rule-Based Reasoner.
6. Calculate confidence score.
7. Invoke Gemini only when required.
8. Generate final routing decision.
9. Export predictions to CSV.
10. Evaluate predictions.

---

# Routing Actions

| Action | Description |
|---------|-------------|
| Notify | Urgent or important messages |
| Digest | Informational messages for later review |
| Mute | Spam, promotions, advertisements, scams |

---

# Message Types Supported

- Banking
- Payment Reminder
- OTP
- Delivery Update
- Calendar Reminder
- Meeting
- Education
- Family
- Friends
- Government
- Emergency
- Promotion
- Newsletter
- Advertisement
- Spam
- Scam

---

# Hybrid AI Design

The project follows a hybrid reasoning approach.

### Rule-Based Reasoner

Uses:

- Business Verification
- Sender Trust
- Domain Matching
- Historical Evidence
- User Engagement
- Notification Dismiss History
- Group Metadata
- Forward Count
- User-Business History
- DND Signals

to generate:

- Action
- Message Type
- Confidence
- Human-readable Reason

---

### Gemini AI

Gemini is invoked only when:

- Confidence is low
- Image messages are detected
- Voice notes are detected
- Conflicting routing signals exist

If Gemini is unavailable or quota is exceeded, the system automatically falls back to the Rule-Based Reasoner.

---

# Confidence Scoring

Confidence is dynamically computed using weighted signals such as:

- Sender Trust
- Historical Evidence
- Business Verification
- User Engagement
- Forward Count
- Spam Indicators
- Urgency
- Group Context
- Media Analysis

The score ranges from **0.0 to 1.0**.

---

# Evaluation Metrics

The evaluator reports:

- Action Accuracy
- Message Type Accuracy
- Overall Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Current verified results:

| Metric | Score |
|---------|-------|
| Action Accuracy | **0.667** |
| Message Accuracy | **0.567** |

---

# Tech Stack

- Python 3.10+
- Pandas
- Google Gemini API
- CSV Processing
- Rule-Based AI
- Logging

---

# Installation

Clone the repository.

```bash
git clone https://github.com/charishmap3/Orchestrate-hackathon.git
cd Orchestrate-hackathon
```

Install dependencies.

```bash
pip install -r code/requirements.txt
```

Create a `.env` file.

```
GEMINI_API_KEY=YOUR_API_KEY
```

Run the project.

```bash
python code/main.py
```

---

# Output

The generated predictions are saved as:

```
dataset/output.csv
```

The output contains:

- Message ID
- Action
- Message Type
- Reason
- Confidence
- Evidence Message IDs

---

# Sample Console Output

```
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

Output File        : dataset/output.csv

==================================================
```

---

# Future Improvements

- Upgrade to the latest Google GenAI SDK
- Improve multimodal reasoning
- Real-time notification routing
- Reinforcement learning from user feedback
- Multi-agent orchestration
- Web dashboard for monitoring

---

# Author

**Charishma P**

AI & Machine Learning Engineering Student

Hackathon Project – Intelligent Message Routing using Hybrid AI

---

# License

This project is licensed under the MIT License.
