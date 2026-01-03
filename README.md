# GRPO vs DPO: Training LLMs for Step-by-Step Reasoning

A comparison of two reinforcement learning optimization methods/approaches for fine-tuning language models to use step-by-step reasoning on math problems.

## Overview

This project fine-tunes **Qwen 2.5-0.5B** on the **GSM8K** dataset using two different RL methods:

| Method | Description |
|--------|-------------|
| **GRPO** | Group Relative Policy Optimization - uses a reward function during training |
| **DPO** | Direct Preference Optimization - learns from preference pairs (chosen vs rejected) |

## Results

Training on 100 examples, evaluating on 30:

| Model | Accuracy | Avg Steps |
|-------|----------|-----------|
| Baseline | 16.7% | 2.77 |
| GRPO | 13.3% | 3.60 |
| **DPO** | **30.0%** | 0.03 |
