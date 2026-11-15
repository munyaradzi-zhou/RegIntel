"""Deterministic guards around the model: they run in code, so the model cannot skip them.

- `screen_question`: blocks instruction-override and prompt-extraction attempts in the question.
- `screen_passages`: drops retrieved or tool text that carries instructions (indirect injection).

They are a first layer, not the only one: the answer verifier independently prevents an injected
claim from being shown unless it is quoted from real evidence, and the evaluation measures each
layer's contribution (tests/evaluation/evaluate_injection.py).
"""
