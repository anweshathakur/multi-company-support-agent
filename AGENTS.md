# AGENTS.md

## Overview

This project implements an AI Support Triage Agent that automatically classifies and resolves customer support tickets across multiple companies (HackerRank, Claude, Visa).

## Core Capabilities

- Company detection from ticket text
- Request type classification (bug, feature, invalid, product issue)
- Product area routing
- Semantic retrieval using embeddings (SentenceTransformers)
- Response generation using rule-based + RAG hybrid approach
- Escalation handling for critical issues

## Architecture

- Embedding Model: sentence-transformers (MiniLM)
- Retrieval: cosine similarity over support corpus
- Classification: hybrid keyword + semantic rules
- Response engine: template-based + contextual grounding

## Entry Point

Run the system using:

```bash
python code/main.py
