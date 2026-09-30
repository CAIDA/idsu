# Adapting IDSU to another field

No code in `idsu/` or `bin/` is specific to Internet measurement. The field is carried by
three inputs: the **target list** of resources to look for, a set of **papers**, and a
**ground truth** saying which resources each paper mentions and uses. A new field needs
its own version of each, plus a config file that points at them.

## 1. Lay out the corpus

Copy the shape of `data/imc2023/` ([its README](../data/imc2023/README.md) documents
every file):

```
data/<corpus>/
  <paper>.pdf ...
  papers_training.yaml  papers_validation.yaml  papers_evaluation.yaml
  _metadata/
    target-list.yaml
    ground-truth.yaml
    papers.yaml            # written by bin/build_papers_manifest.py
```

Then copy `config.yaml` to `config-<corpus>.yaml` and change the `data:` paths. Every tool
takes `--config config-<corpus>.yaml`.

## 2. Write the target list

One entry per resource, with `id`, `name`, `description`, `type` (`dataset` or `tool`),
`urls` and `references`. The whole list goes into the prompt, so these fields are how the
model recognises a resource:

- **`id`** is what extraction returns and what the ground truth must use. `Org|Resource`
  keeps ids unique across providers.
- **`description`** should say what the resource is and how it appears in papers.
- **`references`** should give the way papers usually cite it. A citation to the paper
  that introduced a resource counts as a mention even when its name never appears in the
  text.

The list is the scope of the answer: the prompt tells the model to extract only resources
on it, however prominent another resource is in a paper.

## 3. Annotate the ground truth

For each paper, list every target-list resource it mentions, with verbatim `quotes` from
the paper, each marked `used_in_paper: true` (used in the paper's own work) or `false`
(only mentioned), and the paper's bibliography entries for it. Include papers with no
target-list resources, as `resources: []`: anything extracted from them and marked used
counts as a false positive.

Check the annotations against the papers:

```bash
uv run bin/check_quotes.py --gt data/<corpus>/_metadata/ground-truth.yaml --papers-dir data/<corpus>
```

It lists every quote it cannot find in its paper's extracted text. Training papers'
quotes are shown to the model as few-shot examples, so a wrong one teaches it the wrong
thing.

## 4. Split the papers

List each paper in exactly one of `papers_training.yaml`, `papers_validation.yaml` and
`papers_evaluation.yaml`, as `- filename: <pdf name>` (the `category` field is optional).
Training papers do two jobs: they are the pool that few-shot examples are drawn from, and
the split you tune on. Keep evaluation papers out of any decision until the end.

Then record each paper's identifier and hash:

```bash
uv run bin/build_papers_manifest.py --papers-dir data/<corpus> --out data/<corpus>/_metadata/papers.yaml
```

## 5. Review the prompt

The extraction prompt is `idsu/prompts/extraction-system.prompt`. Its rules are about
mentions and usage in general, but three of its examples come from Internet measurement:
ZMap in rule 4, Censys in rule 5 and Tor Metrics in rule 8. Swapping them for examples
from the new field is optional, and it changes the prompt. Every run records a hash of
each prompt file in its `run.yaml`, so runs made with different prompts can be told apart.

## 6. Run and score

```bash
uv run bin/extract.py --config config-<corpus>.yaml --split training
uv run bin/evaluate.py --config config-<corpus>.yaml runs/<dir>
```

[REPRODUCE.md](REPRODUCE.md) covers the options: dry runs, resuming, fuzzy-only scoring
and other models.

## Current limit

`bin/extract.py` is a benchmark driver. Every paper it runs on must be in the ground truth,
and it stops if one is not. Running it on papers nobody has annotated is not supported yet.
