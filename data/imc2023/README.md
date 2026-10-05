# The IMC 2023 benchmark

45 papers from the ACM Internet Measurement Conference 2023, annotated for which resources
from an 83-item target list each paper mentions, and whether it used them. The PDFs are
not redistributed: `uv run bin/fetch_papers.py` puts them in this directory and checks
each against its recorded hash.

## Files the pipeline reads

All paths are set in `config.yaml` under `data:`.

| File | What |
|---|---|
| `_metadata/target-list.yaml` | The 83 resources extraction is restricted to |
| `_metadata/ground-truth-merged.yaml` | The reference ground truth: every reported number is scored against it |
| `_metadata/ground-truth.yaml` | The earlier ground truth (2026-04-06), before the corrections; `bin/evaluate.py --ground-truth baseline` |
| `_metadata/papers.yaml` | One record per paper: filename, split, DOI or arXiv id, SHA-256 |
| `papers_training.yaml`, `papers_validation.yaml`, `papers_evaluation.yaml` | The splits: 25 / 10 / 10 papers, disjoint |

Other files in this directory are earlier working copies. Nothing in `idsu/` or `bin/`
reads them.

## Target list

A single YAML list. Each entry:

| Field | What |
|---|---|
| `id` | The identifier extraction must return, usually `Org\|Resource` (`CAIDA\|AS Rank`); a few have no org prefix (`BannerClick`) |
| `name` | Display name |
| `description` | What the resource is; shown to the model |
| `type` | `dataset` (50), `tool` (32) or `data` (1, same as `dataset`) |
| `urls`, `references` | Where the resource lives, and how it is usually cited; shown to the model |

The whole list goes into the extraction prompt, so its descriptions and references are
what the model has to recognise a resource by.

## Ground truth

A YAML stream with one document per paper, separated by `---`; read it with
`yaml.safe_load_all()`.

```yaml
filename: 2023-sander-quic-ecn.pdf
resources:
- id: CAIDA|AS2Org             # a target-list id
  name: CAIDA|AS2Org
  type: dataset                # or tool
  urls:
  - https://www.caida.org/catalog/datasets/as-organizations/
  quotes:                      # verbatim from the paper
  - quote: The AS organizations are inferred via CAIDA's as2org dataset [25] ...
    used_in_paper: true        # true: used in the paper's own work; false: only mentioned
  references:                  # the paper's bibliography entries for it
  - '[25] CAIDA. 2023. The CAIDA UCSD AS to Organization Mapping Dataset, April 2023.'
```

- A resource counts as **used** in a paper if any of its quotes is marked used.
- Every `id` is on the target list, and no paper lists an id twice.
- Six papers have no target-list resources (`resources: []`); a correct run extracts
  nothing from them.

`ground-truth-merged.yaml` holds 165 resource instances over the 45 papers:

| Split | Papers | Instances | Used | Mentioned only |
|---|---|---|---|---|
| Training | 25 | 92 | 69 | 23 |
| Validation | 10 | 32 | 29 | 3 |
| Evaluation | 10 | 41 | 34 | 7 |

`bin/check_quotes.py` looks for every quote in its paper's extracted text: 225 of the 232
are found, and it lists the 7 that are not, each with its best fuzzy-match score.

## Splits and categories

Each split file lists `filename` and `category`. The category was assigned when the
splits were built, from the fuller annotations of the time, which also covered resources
outside the target list. It exists to balance the splits and is not recomputed when the
ground truth changes. In order of precedence:

| Category | The paper had |
|---|---|
| `used in list` | at least one used target-list resource |
| `unused in list` | target-list resources, none used |
| `used out list` | no target-list resources, at least one other used resource |
| `unused out list` | only other resources, none used |
| `none` | no annotated resources |

Training papers are also the pool that few-shot examples are drawn from; a paper is never
its own example.

## papers.yaml

Written by `bin/build_papers_manifest.py`, which reads each PDF's DOI or arXiv stamp.

```yaml
- filename: Does It Spin- On the Adoption and Use of QUICs Spin Bit.pdf
  split: evaluation
  category: used in list
  doi: 10.1145/3618257.3624844
  arxiv: 2310.02599v1          # the exact version the annotations were made against
  sha256: 8e7008…
```

One paper, "On the Similarity of Web Measurements Under Different Experimental Setups",
prints no usable identifier. `bin/fetch_papers.py` asks for it by title.
