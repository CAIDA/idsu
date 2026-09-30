# Methods

How the pipeline extracts and scores, and what the project tried on the way to it. The
code is `idsu/`; each stage names the module that implements it.

## Extraction (`idsu/extract.py`)

One model call per paper per run. The messages are:

1. **System prompt** (`idsu/prompts/extraction-system.prompt`): the rules, then the whole
   target list. It is the "unified" prompt, which merged Nathan Man's and Ryan Ding's
   revisions. Its rules cover:
   - what counts as a mention: a name, an alias, or a citation of the paper that
     introduced the resource;
   - that an organisation's name alone is not a mention of its resources;
   - what counts as use. Use as a baseline, comparison or ground truth counts. Citing a
     resource for a statistic, or for ethics guidance it published, does not.
2. **Few-shot examples**: by default one training paper that has target-list resources,
   never the paper being extracted. With more shots, half come from training papers
   without any (the odd one goes to papers with). In
   `partial` mode the example gives that paper's ground-truth quotes and references and
   then its ground-truth resources as the answer. In `whole` mode the same material is
   labelled a complete worked example.
3. **The paper**: its full text, extracted with pypdf.

The model must answer with a JSON object of `resources`, each with `id`, `name`, `type`,
`used_in_paper_reasoning`, `used_in_paper`, `quotes` and `references`. The reasoning field
comes before the label so the model states its evidence first. Calls use JSON mode at
temperature 0, and Gemma is served with its thinking mode on. The response is parsed with
`json-repair`, and a resource is dropped if the model marked it `in_target_list: false` or
gave no quote.

## Scoring (`idsu/match.py`, `idsu/metrics.py`, `idsu/evaluate.py`)

### Matching extractions to the ground truth

Each extracted resource is paired with at most one ground-truth resource of the same paper,
and each ground-truth resource with at most one extraction:

1. an exact `id` match;
2. otherwise the best fuzzy `id` match (thefuzz `ratio` ≥ 80), or the best
   `references` match (`token_sort_ratio` ≥ 80), whichever scores higher. When several
   ground-truth resources tie on references, the `id` score decides;
3. anything still unmatched goes to an LLM judge (`idsu/prompts/matching-judge.prompt`),
   which may pair it with a remaining ground-truth resource or with none.

`bin/evaluate.py --no-llm` stops after step 2.

### The metric: "extraction + used"

The positive class is **used in the paper**. The ground truth is treated as useful but not
exhaustive, so an extraction the model correctly calls unused is not penalised just
because the ground truth lacks it:

| Extracted | Model says used | In ground truth | Ground truth says used | Counts as |
|:---:|:---:|:---:|:---:|:---:|
| yes | yes | yes | yes | TP |
| yes | yes | yes | no | FP |
| yes | yes | no | — | FP |
| yes | no | yes | yes | FN |
| yes | no | yes | no | TN |
| yes | no | no | — | TN |
| no | — | yes | yes | FN |
| no | — | yes | no | TN |

Counts are summed over all papers in a run, accuracy, precision, recall and F1 are
computed from the sums, and the reported figure is the mean and sample SD over runs. The
scorer also writes two secondary matrices: matched vs unmatched regardless of the label,
and agreement on the label over matched pairs only.

### Differences from the scorer behind the reported numbers

The release scorer is a port of `extraction-eval-avg-new.py` from the project's history.
On the same extractions and the same judge output it produces identical confusion
matrices; that was checked on synthetic runs. It differs in three ways:

- **No human step.** The original showed every proposed match to a person to approve or
  correct before counting it. The release accepts the automatic matches.
  `--use-saved-matches` scores approved `matched-resources.yaml` files where they exist.
- **Stops instead of skipping.** A batch with no ground truth, two batches for one paper,
  or a paper missing from a run now stops the scorer. The original skipped each of these
  without saying so.
- **Reports what it leaves out.** Both scorers score only extracted resources that carry
  an `in_target_list` field. The prompt's output schema does not ask for that field, so
  whether a resource is scored depends on whether the model added it anyway. The release
  keeps the rule, for comparability, and reports the count per run under "Unscored".

## What was tried

The final pipeline came out of these studies. The ones by Nathan Man are from his internal
reports (2026); the full reports, code and run outputs are in the project's GitLab
history.

- **Reasoning field** (Nathan Man). Adding `used_in_paper_reasoning` to the output raised
  training-set F1 from 0.8697 (SD 0.0176) to 0.8783 (SD 0.0137) over 5 runs each. That is
  within run-to-run spread, but it fixed specific papers: a resource missed in 4 of 5 runs
  was found in all 5. The field is in the final prompt.
- **Two passes** (Nathan Man). A first call to extract and a second to label use scored F1
  0.897 (SD 0.0066) against 0.9148 (SD 0.0045) for a single call, on Gemma. It was dropped
  as more complex and no better.
- **Matching through the target list** (Nathan Man). On 25 training papers, every
  extracted resource matched a target-list entry, and routing matches through the list
  lost none of the 71 ground-truth matches out of 78 extractions.
- **Unified prompt and corrected ground truth** (Nathan Man, Ryan Ding). Merging their
  prompt revisions and reconciling ground-truth corrections gave the reported results: F1
  0.9363 on training and 0.9177 held out.
- **Fuzzy thresholds and fuzzy vs LLM matching** (Nathan Man). A sweep of thresholds 0–100
  against hand-approved matches found 80 best for both `ratio` on ids and
  `token_sort_ratio` on references. At those thresholds, fuzzy and LLM matching (Gemma)
  produced identical confusion matrices on 80 training-set resources (F1 0.9936). The
  pipeline therefore matches fuzzily first and keeps the LLM for what is left.
- **Number and kind of shots** (Jenny Xu, Qwen3, 25 training papers, an earlier prompt).
  More shots did not help: F1 was 0.666 with one partial shot, 0.642 with two and 0.578
  with three. One whole-paper shot scored 0.671 but used about 63% more prompt tokens than
  one partial shot. The pipeline uses one partial shot.
- **Several model families** (Ryan Ding). A harness ran Gemma, GLM, Kimi and Qwen,
  5 runs each, with its own prompt and matcher. Its numbers are not comparable with the
  ones above, and it is not part of this release. `bin/benchmark_models.py` compares
  models on this pipeline instead.

## The proposal and this release

The project's proposal ([summary](https://www.caida.org/funding/eager-idsu/),
[full text](https://www.caida.org/funding/eager-idsu/eager-idsu_proposal.pdf)) set out a
prompt-engineering method for its resource-utility task. The table maps each part to the
release.

| The proposal | In this release |
|---|---|
| A training set of manually labelled resources: URL, type, mentioned or used, the sentences behind the label, and the references those sentences cite | The ground truth: `urls`, `type`, `quotes` with `used_in_paper`, `references` ([data/imc2023/README.md](../data/imc2023/README.md)) |
| Few-shot examples built from a labelled paper's sentences and nearby references, with the full paper as the ideal but costly alternative | `partial` and `whole` shots (`extraction.shot_mode`) |
| Negative examples, papers that cite a resource without using it, to limit false positives | A shot's answer includes the resources its paper only mentions, labelled `used_in_paper: false`. With two or more shots, half are papers with no target-list resources at all |
| Trade-offs between accuracy and cost across numbers of shots | The shot-count study above, and `extraction.shots` (1–3) |
| Chain-of-thought | The `used_in_paper_reasoning` field, and Gemma's thinking mode |
| A final query with the target paper's full text | The last message of every prompt |
| Repairing the response's JSON with json-repair | `idsu/extract.py` parses every response through `json-repair` |
| Flagging responses whose sentences or references are not in the paper | Not implemented for model output. `bin/check_quotes.py` does this check for the ground truth |
| Evaluation against the manually labelled papers | `bin/evaluate.py` and the "extraction + used" metric |
| LLaMA 3.1-70B on SDSC LLM | The reported results use Gemma on the NRP endpoint; `bin/benchmark_models.py` compares other models |
| Fine-tuning, and if time allowed, RAG and prompt tuning | Not in this release. The project's history holds no experiments with them |
| Instructions and code for adaptation by other disciplines | [ADAPTING.md](ADAPTING.md) |
| Integration into CAIDA's catalog | Not part of this release. `bin/extract.py` runs only on papers that have a ground truth |

## Ground truth

The annotations were built by Bradley Huffaker from December 2025 to April 2026, as
`_metadata/ground-truth.yaml` (139 instances). In late May 2026 Nathan Man and Ryan Ding
corrected them, adding missing used resources and fixing labels. In June Nathan merged
both sets of corrections into `_metadata/ground-truth-merged.yaml`. Its ids were
aligned with the target list for the release, and one leftover duplicate (a paper that
examined ZMap and rejected it) was removed, leaving 165 instances. The earlier file stays
available: `bin/evaluate.py --ground-truth baseline` scores the same runs against it.
