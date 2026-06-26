"""Feedback-driven data generation pipeline for probe training.

Stages:
  1. collect_errors  - Gather FP/FN cases from OOD bloom eval + neutral validation
  2. analyze_errors  - Multi-agent Opus 4.6 analysis of error patterns
  3. generate_data   - Sonnet 4.6 transcript generation from analysis suggestions
  4. merge_data      - Combine new transcripts with existing training data
"""
