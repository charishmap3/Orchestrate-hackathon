# Orchestrate Phase 1

This repository contains the Phase 1 implementation for the HackerRank Orchestrate challenge.

## Purpose

- Load and validate the dataset from `dataset/`
- Build a routing context for a single message
- Retrieve historical evidence from message history and event logs

## Structure

- `code/agents/loader.py` - dataset loader and validator
- `code/agents/context_builder.py` - builds `RoutingContext` from related dataset tables
- `code/agents/retrieval.py` - finds historical evidence messages and related events
- `code/models/schemas.py` - typed dataclasses for catalog and context objects
- `code/main.py` - CLI entrypoint for inspection

## Requirements

- Python 3.10+
- `pandas`

Install dependencies:

```bash
python -m pip install -r code/requirements.txt
```

## Run

```bash
python code/main.py --message-id msg_023
```

Use `--verbose` for debug logging.
