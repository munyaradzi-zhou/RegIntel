"""Train the ML router, track the run in MLflow, register it, and export the served JSON model.

    python scripts/train_router.py                       # train, evaluate, register, export router-v<N>.json
    python scripts/train_router.py --no-register         # train and evaluate only (nothing exported)
    python scripts/train_router.py --llm-results tests/evaluation/results/<label>/router.json

Retraining is the same command after the dataset changes (scripts/build_router_data.py, or new
labelled questions added to router_dataset.json): a new run, a new registered version, a new file.

Steps:
1. Vocabulary and idf from the training split only (min_df 2, at most MAX_FEATURES by document
   frequency); features from regintel.ml_router.features, the function the API uses.
2. C chosen by 5-fold cross-validation grouped by parent question, so a question and its
   paraphrases are never on both sides of a fold. Score: macro-F1.
3. Fallback thresholds, one per predicted route, from the out-of-fold probabilities: the lowest
   threshold at which that route's confident predictions reach TARGET_PRECISION. A data question
   is easy to recognise; "one section or several" is not, so one global threshold either trusts
   the hard cases or sends nearly everything to the LLM.
4. Test: the held-out real questions only, against the LLM router's decisions on the same
   questions. Unanswerable questions count as correct when not sent to the agent (as in
   evaluate_router.py).
5. MLflow (local SQLite store in mlruns/): params, metrics, the confusion matrix and the model
   file; the model is registered as "regintel-router" and exported with that version number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from regintel.ml_router import MODELS, RouterModel, features, raw_features

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests" / "evaluation" / "datasets" / "router_dataset.json"
MLRUNS = ROOT / "mlruns"
CLASSES = ["agent", "guidance_multi", "guidance_single"]
MIN_DF = 2
MAX_FEATURES = 6000
C_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
TARGET_PRECISION = 0.95  # per route, out of fold: about the LLM router's own accuracy (0.96), fixed before testing
FOLDS = 5


def vocabulary(texts: list[str]) -> tuple[dict[str, int], list[float]]:
    df: Counter[str] = Counter()
    for text in texts:
        df.update(raw_features(text).keys())
    kept = sorted((g for g, n in df.items() if n >= MIN_DF), key=lambda g: (-df[g], g))[:MAX_FEATURES]
    kept.sort()
    n = len(texts)
    return {g: i for i, g in enumerate(kept)}, [math.log((1 + n) / (1 + df[g])) + 1.0 for g in kept]


def matrix(texts: list[str], vocab: dict[str, int], idf: list[float]) -> csr_matrix:
    rows, cols, vals = [], [], []
    for r, text in enumerate(texts):
        for c, v in features(text, vocab, idf).items():
            rows.append(r)
            cols.append(c)
            vals.append(v)
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)))


def correct(route: str, predicted: str, kind: str) -> bool:
    return predicted != "agent" if kind == "unanswerable" else predicted == route


def fit(x: csr_matrix, y: list[str], c: float) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(x, y)


def cross_validate(x: csr_matrix, y: list[str], groups: list[str], c: float) -> np.ndarray:
    """Out-of-fold probabilities (columns in CLASSES order)."""
    oof = np.zeros((len(y), len(CLASSES)))
    for train, test in GroupKFold(n_splits=FOLDS).split(x, y, groups):
        model = fit(x[train], [y[i] for i in train], c)
        oof[test] = model.predict_proba(x[test])[:, [list(model.classes_).index(k) for k in CLASSES]]
    return oof


def choose_thresholds(oof: np.ndarray, y: list[str]) -> dict[str, float]:
    top, predicted = oof.max(axis=1), [CLASSES[i] for i in oof.argmax(axis=1)]
    thresholds = {}
    for route in CLASSES:
        thresholds[route] = 1.0  # never trusted unless some threshold reaches the target
        for t in np.arange(0.34, 1.0, 0.01):
            chosen = [y[i] == route for i in range(len(y)) if predicted[i] == route and top[i] >= t]
            if chosen and np.mean(chosen) >= TARGET_PRECISION:
                thresholds[route] = round(float(t), 2)
                break
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-register", action="store_true", help="evaluate only; export nothing")
    parser.add_argument("--llm-results", type=Path,
                        default=ROOT / "tests/evaluation/results/2026-11-21-router-v2/router.json")  # fmt: skip
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    raw = DATASET.read_bytes()
    data = json.loads(raw)["questions"]
    train = [q for q in data if q["split"] == "train"]
    test = [q for q in data if q["split"] == "test"]
    vocab, idf = vocabulary([q["question"] for q in train])
    x_train = matrix([q["question"] for q in train], vocab, idf)
    y_train = [q["route"] for q in train]
    groups = [q.get("parent", q["id"]) for q in train]

    cv = {}
    for c in C_GRID:
        oof = cross_validate(x_train, y_train, groups, c)
        cv[c] = f1_score(y_train, [CLASSES[i] for i in oof.argmax(axis=1)], average="macro")
        print(f"C={c:<5} cv macro-F1 {cv[c]:.3f}")
    best_c = max(cv, key=lambda c: (round(cv[c], 3), -c))  # ties: the stronger regularisation
    oof = cross_validate(x_train, y_train, groups, best_c)
    threshold = choose_thresholds(oof, y_train)
    model = fit(x_train, y_train, best_c)
    order = [list(model.classes_).index(k) for k in CLASSES]

    body: dict[str, Any] = {
        "version": 0,
        "classes": CLASSES,
        "vocabulary": vocab,
        "idf": [round(v, 6) for v in idf],
        "coef": [[round(float(v), 6) for v in model.coef_[i]] for i in order],
        "intercept": [round(float(model.intercept_[i]), 6) for i in order],
        "threshold": threshold,
    }
    served = RouterModel(0, CLASSES, vocab, body["idf"], body["coef"], body["intercept"], threshold, {}, "")

    llm: dict[str, str] = {}
    if args.llm_results.exists():
        llm = {r["id"]: r["got"] for r in json.loads(args.llm_results.read_text(encoding="utf-8"))["rows"]}
    started = time.perf_counter()
    predictions = [served.predict(q["question"]) for q in test]
    ms_per_question = (time.perf_counter() - started) * 1000 / len(test)
    rows = []
    for q, (route, p) in zip(test, predictions, strict=True):
        llm_route = llm.get(q["id"])
        fallback = p < threshold[route]
        combined = (llm_route or route) if fallback else route
        rows.append({"id": q["id"], "kind": q["kind"], "route": q["route"], "ml": route, "p": round(p, 3),
                     "llm": llm_route, "fallback": fallback, "combined": combined})  # fmt: skip

    def accuracy(key: str) -> float:
        scored = [r for r in rows if r[key] is not None]
        return round(sum(correct(r["route"], r[key], r["kind"]) for r in scored) / len(scored), 3)

    def macro_f1(key: str) -> float:
        scored = [r for r in rows if r[key] is not None and r["kind"] != "unanswerable"]
        return round(float(f1_score([r["route"] for r in scored], [r[key] for r in scored], average="macro")), 3)

    metrics = {
        "cv_macro_f1": round(cv[best_c], 3),
        "test_accuracy_ml": accuracy("ml"),
        "test_macro_f1_ml": macro_f1("ml"),
        "test_accuracy_ml_with_fallback": accuracy("combined") if llm else None,
        "test_accuracy_llm": accuracy("llm") if llm else None,
        "test_macro_f1_llm": macro_f1("llm") if llm else None,
        "test_fallback_rate": round(sum(r["fallback"] for r in rows) / len(rows), 3),
        "ms_per_question_ml": round(ms_per_question, 3),
    }
    matrix_ml = confusion_matrix([r["route"] for r in rows], [r["ml"] for r in rows], labels=CLASSES).tolist()
    report = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "train": len(train),
        "train_real": sum(q["source"] == "real" for q in train),
        "test": len(test),
        "C": best_c,
        "features": len(vocab),
        "threshold": threshold,
        "metrics": metrics,
        "confusion_ml": {"labels": CLASSES, "rows_true_cols_predicted": matrix_ml},
        "test_rows": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "test_rows"}, indent=1))
    for r in rows:
        if not correct(r["route"], r["ml"], r["kind"]) or r["fallback"]:
            print(f"  {r['id']:8} true={r['route']:16} ml={r['ml']:16} p={r['p']:.2f} llm={r['llm']}")
    if args.no_register:
        return

    import mlflow
    import mlflow.pyfunc

    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("regintel-router")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "C": best_c, "min_df": MIN_DF, "max_features": MAX_FEATURES, "features": len(vocab),
            "thresholds": json.dumps(threshold), "target_precision": TARGET_PRECISION, "folds": FOLDS,
            "train": len(train), "test": len(test), "dataset_sha256": report["dataset_sha256"],
        })  # fmt: skip
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.log_dict(report, "evaluation.json")
        draft = MLRUNS / "router-draft.json"
        draft.write_text(json.dumps(body), encoding="utf-8")
        info = mlflow.pyfunc.log_model(
            name="router",
            python_model=ServedRouter(),
            artifacts={"model": str(draft)},
            registered_model_name="regintel-router",
        )
        version = int(info.registered_model_version)
        body["version"] = version
        body["meta"] = {"mlflow_run_id": run.info.run_id, "trained_at": time.strftime("%Y-%m-%d"),
                        "dataset_sha256": report["dataset_sha256"], "C": best_c, "metrics": metrics,
                        "confusion_ml": report["confusion_ml"]}  # fmt: skip
        MODELS.mkdir(exist_ok=True)
        out = MODELS / f"router-v{version}.json"
        out.write_text(json.dumps(body, separators=(",", ":")) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(out))
        (ROOT / "tests" / "evaluation" / "results").mkdir(exist_ok=True)
        print(f"registered regintel-router version {version}; exported {out.relative_to(ROOT)}")


class ServedRouter:  # an MLflow pyfunc around the exported JSON, so the registry can serve it too
    def load_context(self, context: Any) -> None:
        self.model = RouterModel.load(Path(context.artifacts["model"]))

    def predict(self, context: Any, model_input: Any) -> list[str]:
        return [self.model.predict(str(text))[0] for text in list(model_input.iloc[:, 0])]


if __name__ == "__main__":
    import mlflow.pyfunc

    ServedRouter = type("ServedRouter", (mlflow.pyfunc.PythonModel,), dict(ServedRouter.__dict__))  # type: ignore[misc]
    main()
