# Evaluation Suite

This directory contains the versioned, deterministic regression dataset and its
accepted baseline. It is designed to run locally and in CI without API keys or
model usage charges.

## What it measures

- FAISS retrieval hit rate, mean reciprocal rank, and concept coverage using
  deterministic local embeddings
- Tool-selection accuracy from curated representative traces
- Claim grounding and citation completeness from curated generated answers
- Tool success rate from representative tool events

The retrieval checks execute the project's FAISS retrieval path. Tool-selection
and generation checks use curated fixtures, including known failures, so that
regression results remain repeatable and do not imply perfect model behavior.
They should be extended periodically with reviewed staging or production traces.
This suite is a regression gate; it does not replace live-model evaluation or
production monitoring.

The challenge set includes paraphrased queries, lexically similar distractor
documents, a no-tool request, an ambiguous tool-routing failure, an unsupported
claim, and an incomplete citation. Known failures remain visible in the report
and are protected by both absolute quality thresholds and baseline tolerances.

## Run the regression gate

```bash
uv run python scripts/run_evaluation.py
```

The command writes machine-readable and human-readable reports to:

- `output/evaluations/latest.json`
- `output/evaluations/latest.md`

It exits with status `1` if a metric falls below its absolute minimum or drops
too far below the checked-in baseline.

## Update the baseline

Only update the baseline after reviewing an intentional model, prompt, tool, or
dataset change:

```bash
uv run python scripts/run_evaluation.py --update-baseline
```

Commit `dataset.json` and `baseline.json` together when the expected behavior
changes. Never update the baseline merely to make a failing regression pass.
