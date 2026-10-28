# Applying NAIRR Pilot Resources to Infer Dataset Utility (IDSU)

## Which datasets and software does a paper use?

IDSU reads a research paper and reports which resources from a fixed **target list** of
datasets and software tools the paper mentions, and for each one whether the authors
actually **used** it or only cited it. It runs an instruction-tuned LLM through any
OpenAI-compatible API, and it comes with an annotated benchmark: 45 papers from ACM IMC
2023, scored against an 83-resource target list of Internet-measurement datasets and tools.

It is part of CAIDA's project [**Applying NAIRR Pilot Resources to Infer Data Set
Utility**](https://www.caida.org/funding/eager-idsu/) (NSF OAC-2526448). The project's goal is a service that infers the utility of datasets and
software tools from their documented use in scientific publications.

## Quick start

Follow these steps for a first run. For the full reported results, continue with
[REPRODUCE.md](docs/REPRODUCE.md) after setup. Run commands from the repository root.

### 1. Set up

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14, plus a key for an
OpenAI-compatible endpoint.

```bash
cp .env.example .env              # then fill in OPENAI_API_KEY and OPENAI_BASE_URL
uv run bin/fetch_papers.py        # the 45 benchmark PDFs, checked by hash
```

`bin/fetch_papers.py` downloads the arXiv papers itself and lists the rest with their DOI
links. Download those in a browser, then run `uv run bin/fetch_papers.py --import <dir>`.
Model names and other defaults are in `config.yaml`. For another endpoint, see [model setup](docs/REPRODUCE.md#1-setup).

### 2. Check the setup (optional, no API calls)

```bash
uv run bin/extract.py --split validation --only "[paper_name].pdf" --runs 1 --dry-run --out runs/dry
```

Replace `[paper_name].pdf` with a filename from `data/imc2023/papers_validation.yaml`.
This writes a prompt and an empty extraction. It checks setup, not model quality.
PDFs are still required, including the training PDFs used as examples.

### 3. Run and score one paper

```bash
uv run bin/extract.py --split validation --only "Localizing Traffic Differentiation.pdf" --runs 1 --out runs/small
uv run bin/evaluate.py runs/small --no-llm
```

Extraction calls the model. Scoring with `--no-llm` uses text matching without API
calls. Read the scores in `runs/small/eval-ground-truth-merged-fuzzy/summary.md`.
Use a new output folder for a new run. To change the run settings, see [extraction options](docs/REPRODUCE.md#extraction-options).

### 4. Run the full comparison

Training uses 25 papers. Held-out combines validation and evaluation (20 papers).
These are paper groups; `--runs` controls how many times extraction is repeated.
The default is 5 runs. The training split also supplies prompt examples; this
script does not train a model.

Follow [REPRODUCE.md](docs/REPRODUCE.md#2-extraction) for full extraction and scoring
commands. That guide also covers scoring against the earlier ground truth,
comparing text matching with the LLM judge, and trying other models.
Use [the summary comparison below](#reported-results) to view split scores together.

## What is here

| Path | What |
|---|---|
| `bin/extract.py` | Extraction: one prompt per paper, run several times |
| `bin/evaluate.py` | Scoring against the ground truth: mean and SD across runs |
| `bin/compare_summaries.py` | Training and validation/held-out summary metrics side by side |
| `bin/benchmark_models.py` | The same pipeline over several models, one comparison table |
| `bin/fetch_papers.py` | The benchmark PDFs, each checked against its recorded SHA-256 |
| `bin/check_quotes.py` | Checks that every ground-truth quote appears in its paper |
| `bin/build_papers_manifest.py` | Rebuilds `papers.yaml` from the PDFs |
| `idsu/` | The library the tools share; the prompts are in `idsu/prompts/` |
| `data/imc2023/` | The benchmark: target list, ground truth, splits ([README](data/imc2023/README.md)) |
| `config.yaml` | Models, paths, thresholds, number of runs |
| `results/imc2023-gemma-20261005/` | The extractions and scores behind the reported results |
| `docs/REPRODUCE.md` | Reproducing the reported numbers, command by command |
| `docs/ADAPTING.md` | Using IDSU on another field: a new target list and ground truth |
| `docs/METHODS.md` | How extraction and scoring work, and what was tried along the way |

## Reported results

Compare existing evaluation summaries without rerunning extraction or scoring:

```bash
uv run bin/compare_summaries.py --training results/imc2023-gemma-20261005/training/eval-ground-truth-merged-llm/summary.md --held-out results/imc2023-gemma-20261005/held-out/eval-ground-truth-merged-llm/summary.md
```

The Markdown table shows accuracy, precision, recall, and F1 as mean (sample SD)
for each split. Paths may also point to evaluation directories containing `summary.md`.
Use `--validation` instead of `--held-out` for a validation split, or supply both
to compare three splits. Add `--output <path>` to also save the table to a file.

The final configuration is the unified single-pass prompt, run on Gemma at temperature 0,
scored against the corrected ground truth. `config.yaml` holds its defaults. The metric is
"extraction + used", as mean and sample SD over 5 runs:

| Split | Papers | Source | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Training | 25 | Internal report | 0.9051 (0.0201) | 0.9434 (0.0161) | 0.9294 (0.0123) | 0.9363 (0.0134) |
| | | This release | 0.9369 (0.0249) | 0.9560 (0.0148) | 0.9588 (0.0192) | 0.9574 (0.0168) |
| | | Reproduction (2026-10-12) | 0.9480 (0.0117) | 0.9567 (0.0100) | 0.9735 (0.0066) | 0.9650 (0.0079) |
| Validation + evaluation (held out) | 20 | Internal report | 0.8640 (0.0239) | 0.9313 (0.0076) | 0.9048 (0.0251) | 0.9177 (0.0152) |
| | | This release | 0.8700 (0.0170) | 0.9188 (0.0148) | 0.9246 (0.0187) | 0.9215 (0.0112) |
| | | Reproduction (2026-10-12) | 0.8619 (0.0165) | 0.9150 (0.0068) | 0.9180 (0.0201) | 0.9165 (0.0114) |

"Internal report" is the project's internal report (N. Man, 2026). "This release" is a
rerun with this code on 2026-10-05 (new extractions, NRP `gemma`, fuzzy matching then the
LLM judge). It also uses a ground truth and target list with one error fixed (see
[docs/METHODS.md](docs/METHODS.md#ground-truth)). The two differ by less than two SDs on
both splits. Its outputs and scores are in
[results/imc2023-gemma-20261005/](results/imc2023-gemma-20261005/).

The reproduction row is a separate five-run execution on 2026-10-12 with the configured
NRP `gemma` and merged ground truth.

Ten of the held-out papers had their ground truth corrected in late May 2026, before the
held-out results were computed. Against the uncorrected ground truth, held-out F1 is 0.8649
(0.0113). How the metric is defined, and how the release scorer relates to the original, is
in [docs/METHODS.md](docs/METHODS.md).

## Credits

Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Terms of use

Use of this repository is subject to CAIDA's
[Acceptable Use Agreement for publicly accessible datasets](https://www.caida.org/about/legal/aua/public_aua/).
The software is licensed for educational, research and non-profit use under the terms in
[LICENSE](LICENSE); commercial use requires permission from UC San Diego's Office of
Innovation and Commercialization.
