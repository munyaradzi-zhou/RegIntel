"""The research agent (Microsoft Agent Framework) and the evidence it gathers.

Flow: the agent decomposes the question and calls tools (guidance search, CDISC lookup, FAERS
computations). Every tool result is recorded as numbered evidence. The answer writer then drafts
statements citing that evidence with verbatim quotes, and the Phase 3 verifier checks them, so a
FAERS count or a CDISC code in an answer must appear in the tool output it cites.
"""
