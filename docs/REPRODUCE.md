# Reproducing the reported results

The numbers in the [README](../README.md#reported-results) come from 5 runs of the
extraction pipeline on the training split and 5 on the held-out papers (validation and
evaluation together), each scored against the reference ground truth. These commands
repeat that. Run them from the repository root.

## 1. Setup

```bash
cp .env.example .env        # OPENAI_API_KEY and OPENAI_BASE_URL
uv run bin/fetch_papers.py  # repeat with --import <dir> until it reports all 45 verified
```

The model names in `config.yaml` (`gemma` for extraction, `gpt-oss` for the matching
judge) are aliases served by the NRP endpoint in `.env.example`. On another endpoint, set
`extraction.model` and `evaluation.judge_model` to the names it serves.

To check the setup without spending calls:

```bash
uv run bin/extract.py --split validation --runs 1 --dry-run --out runs/dry
uv run bin/evaluate.py runs/dry --no-llm
```

The dry run writes every prompt and an empty extraction, so its scores are all zero.

## 2. Extraction

```bash
uv run bin/extract.py --split training                         # 25 papers × 5 runs
uv run bin/extract.py --split validation --split evaluation    # 20 papers × 5 runs
```

Each command prints its output directory, `runs/<timestamp>-<model>-<splits>/`, holding
`run-1` … `run-5`. Each run holds one `batch-*` directory per paper (prompt, raw response,
parsed extraction), plus a `run.yaml` recording the papers, the settings, the git commit
and a hash of each prompt file. A batch that fails after its retries is marked in its
`info.yaml`. Rerun the same command with `--out <that directory> --resume` to retry only
the failures.

Shots are drawn from the training split with a seed derived from `extraction.seed`, the
run number and the paper, so a rerun gives every paper the same examples.

## 3. Scoring

```bash
uv run bin/evaluate.py runs/<training dir>
uv run bin/evaluate.py runs/<held-out dir>
uv run bin/evaluate.py runs/<held-out dir> --ground-truth baseline
```

Each writes `summary.md` (mean and SD table, plus per-run counts), `summary.json` and every
paper's matches to `eval-<ground truth>-<matcher>/` inside the run directory. The third
command scores the same held-out runs against the earlier ground truth, to show how much
of the result depends on the corrections.

`evaluate.py` stops, with no summary, if a paper has no ground truth, a run has two batches
for one paper, or a paper listed in `run.yaml` has no batch. Extracted resources without an
`in_target_list` field are not scored, and each run reports how many there were.

## 4. Matching: fuzzy against the LLM judge

```bash
uv run bin/evaluate.py runs/<dir> --no-llm
```

This scores the same runs with fuzzy matching alone. Comparing it with the default
(fuzzy, then the LLM judge for whatever is left) shows what the judge adds.

## 5. Other models (optional)

```bash
uv run bin/benchmark_models.py --split training
```

This runs the same pipeline once per model in `benchmark.models` and scores each the same
way, with the same judge. Output: `benchmark.md` and `benchmark.json`.

## Recording what you ran

Aliases like `gemma` point at whatever build the endpoint serves that day. Save the model
list alongside any results:

```bash
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models"
```

(`curl` reads those variables from your shell, not from `.env`.)

## What to expect

Temperature is 0, but results still vary from run to run; that spread is what the SD
columns measure, so compare means and SDs rather than single runs. Two differences from
the original setup can move the numbers:

- **Matching.** The original scorer showed every proposed match to a person for approval
  before counting it. The release scorer accepts the fuzzy and LLM matches as they are.
  `--use-saved-matches` scores hand-approved `matched-resources.yaml` files instead, where a
  batch has one.
- **Model builds.** The endpoint may serve a newer build under the same alias.
