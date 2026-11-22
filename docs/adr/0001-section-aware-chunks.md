# 0001. Section-aware chunks over integrated chunking

**Status:** Accepted, 2026-11-13

## Context

Citations must name a section a reader can open (`§4.1.3, p.8`). Azure AI Search's integrated chunking splits PDFs into token windows, which cross section boundaries and carry no locator. Guidance PDFs also have footnotes, running headers, line numbers and garbled formula text.

## Decision

RegIntel chunks the corpus itself: one chunk per section (long sections split), each with its locator, pages and a contextual header naming the document. Footnotes are separated and formula text is repaired. The index stores these chunks; Azure's integrated pipeline is kept only as a comparison.

## Alternatives

- Azure integrated chunking (text split skill): no locators, and measured worse.
- Fixed token windows with overlap: simple, but citations become "page 8, somewhere".

## Consequences

- Retrieval and citation quality depend on code RegIntel owns and tests (chunking, glyph repair and footnote handling have their own unit tests).
- A change to the chunker needs the CI fixture rebuilt (`build_fixtures.py`).

## Evidence

With the same embeddings, planner and reranker, section chunks beat integrated chunking on every retrieval metric: right section first 0.74 vs 0.62 (+0.12 section hit@1, +0.10 hit@5).
