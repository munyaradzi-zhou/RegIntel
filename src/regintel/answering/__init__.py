"""Answer writing with code-enforced grounding.

The model drafts an answer as statements, each citing passages by number with a verbatim quote.
`verify` then checks every quote against the retrieved text and every number against the quotes.
What fails is dropped; an answer left with nothing verified is withheld (fail closed).
"""
