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

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14, plus a key for an
OpenAI-compatible endpoint.

```bash
cp .env.example .env              # then fill in OPENAI_API_KEY and OPENAI_BASE_URL
uv run bin/fetch_papers.py        # the 45 benchmark PDFs, checked by hash
uv run bin/extract.py --split training
uv run bin/evaluate.py runs/<the directory extract.py printed>
```

`bin/fetch_papers.py` downloads the arXiv papers itself and lists the rest with their DOI
links. Download those in a browser, then run `uv run bin/fetch_papers.py --import <dir>`.
Every tool takes `--help`, and every path and parameter is in `config.yaml`.

`bin/extract.py --dry-run` writes every prompt without calling a model, and
`bin/evaluate.py --no-llm` scores with fuzzy matching only, so every step can be tried
without an API key.

## What is here

| Path | What |
|---|---|
| `bin/extract.py` | Extraction: one prompt per paper, run several times |
| `bin/evaluate.py` | Scoring against the ground truth: mean and SD across runs |
| `bin/benchmark_models.py` | The same pipeline over several models, one comparison table |
| `bin/fetch_papers.py` | The benchmark PDFs, each checked against its recorded SHA-256 |
| `bin/check_quotes.py` | Checks that every ground-truth quote appears in its paper |
| `bin/build_papers_manifest.py` | Rebuilds `papers.yaml` from the PDFs |
| `idsu/` | The library the tools share; the prompts are in `idsu/prompts/` |
| `data/imc2023/` | The benchmark: target list, ground truth, splits ([README](data/imc2023/README.md)) |
| `config.yaml` | Models, paths, thresholds, number of runs |
| `docs/REPRODUCE.md` | Reproducing the reported numbers, command by command |
| `docs/ADAPTING.md` | Using IDSU on another field: a new target list and ground truth |
| `docs/METHODS.md` | How extraction and scoring work, and what was tried along the way |

## Reported results

The final configuration is the unified single-pass prompt, run on Gemma at temperature 0,
with the corrected ground truth. `config.yaml` holds the defaults of the script that
produced these numbers. The metric is "extraction + used", as mean and sample SD over 5
runs:

| Split | Papers | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Training | 25 | 0.9051 (0.0201) | 0.9434 (0.0161) | 0.9294 (0.0123) | 0.9363 (0.0134) |
| Validation + evaluation (held out) | 20 | 0.8640 (0.0239) | 0.9313 (0.0076) | 0.9048 (0.0251) | 0.9177 (0.0152) |

These are the numbers from the project's internal report (N. Man, 2026). How the metric
is defined, and how the release scorer relates to the one that produced them, is in
[docs/METHODS.md](docs/METHODS.md).

## Credits

Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Terms of use

Use of this repository is subject to CAIDA's
[Acceptable Use Agreement for publicly accessible datasets](https://www.caida.org/about/legal/aua/public_aua/).
