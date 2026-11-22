# Data card: router dataset

`tests/evaluation/datasets/router_dataset.json`, built by `scripts/build_router_data.py`, used by
`scripts/train_router.py` and by the CI evaluation gate.

## Purpose

To train and test the classifier that picks the answering path for a question: `agent` (data
tools: FAERS, CDISC, openFDA), `guidance_multi` (several regulators, documents or sections) or
`guidance_single` (one focused guidance question).

## Contents

| Split | Source | agent | guidance_multi | guidance_single | Total |
|---|---|---|---|---|---|
| test | real | 5 | 4 | 14 | 23 |
| train | real | 11 | 11 | 31 | 53 |
| train | generated | 77 | 77 | 217 | 371 |

- **Real questions** are the 60 golden questions and 16 tool questions of the evaluation sets,
  written by hand against the corpus and the data tools. The route comes from the question type:
  single and unanswerable → `guidance_single`; comparison and multihop → `guidance_multi`; tool
  questions → `agent`. Unanswerable questions are scored as correct whenever they are not sent to
  the agent, as in the LLM router evaluation.
- **Generated questions** were written by `gpt-5-mini` (2026-11-22) from training questions only:
  4 paraphrases and 3 new questions of the same route for each. Every generated row names its
  `parent`.

## Splits and leakage

The real questions were split once, 70/30 by route, with a fixed seed (`20261122`). The test
split was never shown to the generator and is never used for training or for choosing
hyperparameters. Cross-validation on the training split groups each question with its
paraphrases (by `parent`), so a fold never trains on a paraphrase of a question it is scored on.

## Known limitations

- **Small.** 76 real questions, 23 of them for testing: one question is 4.3 points of test
  accuracy. Treat differences of one or two questions as noise.
- **Generated variants are easier than real traffic.** Paraphrases share their parent's vocabulary,
  and cross-validation on them (macro-F1 0.83) is not a measure of real-world accuracy.
- **"One section or several" is a judgement.** The hardest distinction, single versus multi-part,
  depends on where the guidance puts things; the LLM router also misroutes some multihop questions.
- **English only,** and the regulators, documents and drugs of this corpus. Questions about other
  products or jurisdictions are out of distribution.
- **Labels inherit the golden set's types.** No second annotator; no inter-annotator agreement.

## Maintenance

To retrain, add labelled questions (real ones preferably, with `split: "train"`), or rebuild the
generated variants, then run `python scripts/train_router.py`. Each run is tracked in MLflow
(`mlruns/`), registered as a new version of `regintel-router`, and exported as
`src/regintel/router_models/router-v<N>.json`. Serving picks the version with
`ROUTER_MODEL_VERSION`; the CI gate scores the configured version on the test split.
