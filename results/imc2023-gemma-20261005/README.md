# IMC 2023 corpus, Gemma, 2026-10-05

The runs behind the "This release" rows of the [README](../../README.md#reported-results):
5 runs each on the training split (25 papers) and the held-out papers (validation and
evaluation, 20), made with `bin/extract.py` on the NRP `gemma` endpoint at temperature 0.

| Path | What it holds |
|---|---|
| `<split>/run-<k>/run.yaml` | Papers and settings for that run. `commit` is in the development repository, not this one |
| `<split>/run-<k>/batch-*/extraction-parsed.json` | The model's extracted resources for one paper |
| `<split>/run-<k>/batch-*/info.yaml` | Counts and the shot papers used for that paper |
| `<split>/eval-ground-truth-merged-llm/` | Scored against the corrected ground truth |
| `held-out/eval-ground-truth-llm/` | Scored against the uncorrected ground truth (`--ground-truth baseline`) |
| `models-20261005.json` | The endpoint's model list, saved on 2026-10-05 a few hours before the runs |

The prompts and raw model responses are left out, because they contain paper text. To
rescore without an API key:

```bash
uv run bin/evaluate.py results/imc2023-gemma-20261005/training --no-llm
```

The fuzzy-only scores equal the judged ones in all three summaries: the judge accepted no
match.
