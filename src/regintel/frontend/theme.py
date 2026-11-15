"""Paper & Ember styling for the Streamlit front end.

Colours and fonts are set in .streamlit/config.toml; this module adds what the theme file cannot:
tight serif headings, eyebrows, answer-status pills, source record cards, stat tiles, pastel cards
and the footer. Every string that comes from the API is escaped before it is placed in HTML.
"""

from __future__ import annotations

from html import escape

import streamlit as st

# Data marks: violet (the design system's single "key actor" colour). Validated with the dataviz
# skill's palette checker against the surface: chroma and contrast pass (slate read as grey).
MARK = "#7369AD"
INK = "#14130C"
INK_MUTED = "#5F594A"
RULE = "#D3CEBB"
EMBER_DEEP = "#A14C3F"

_CSS = """
<style>
:root{
  --paper:#E8E3D2; --paper-soft:#EDE8D7; --surface:#FAF4E8; --panel:#E4DDCC;
  --inset:#DED9C6; --rule:#D3CEBB;
  --ink:#14130C; --ink-body:#4F4938; --ink-muted:#5F594A;
  --ember:#C55644; --ember-deep:#A14C3F; --slate:#656456;
  --sage:#B5CAB1; --lilac:#C1BCD6; --peach:#E9C0AA; --on-pastel-body:#36311F;
  --font-serif: Newsreader, Georgia, "Times New Roman", serif;
  --font-sans: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif;
  --font-mono: "JetBrains Mono", Menlo, Consolas, monospace;
}
h1, h2, h3, h4 { font-family: var(--font-serif) !important; font-weight: 400 !important;
                 letter-spacing: -0.02em; line-height: 1.05; color: var(--ink); }
.pe-eyebrow { font: 600 0.8125rem/1.2 var(--font-sans); letter-spacing: 0.15em; text-transform: uppercase;
              color: var(--ember-deep); margin: 0 0 12px; }
.pe-title { font: 400 3rem/1.0 var(--font-serif); letter-spacing: -0.03em; color: var(--ink); margin: 0; }
.pe-title .pe-ember { color: var(--ember); }
.pe-heading { font: 400 2rem/1.05 var(--font-serif); letter-spacing: -0.02em; color: var(--ink); margin: 0 0 8px; }
.pe-sub { font: 400 1rem/1.45 var(--font-sans); color: var(--ink-body); max-width: 46rem; margin: 16px 0 8px; }

/* Answer status: a pill whose colour and words both carry the state */
.pe-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 12px; }
.pe-pill { font: 500 0.8125rem/1 var(--font-sans); border-radius: 999px; padding: 6px 12px; color: var(--ink); }
.pe-pill--grounded { background: var(--sage); }
.pe-pill--partially_grounded { background: var(--peach); }
.pe-pill--declined, .pe-pill--not_grounded { background: var(--inset); }
.pe-pill--blocked { background: var(--slate); color: var(--surface); }
.pe-chip { display: inline-block; font: 400 0.8125rem/1.4 var(--font-mono); background: var(--inset);
           color: var(--ink-body); border-radius: 8px; padding: 2px 10px; }
.pe-answer sup { font: 600 0.7rem var(--font-mono); color: var(--ember-deep); margin-left: 2px; }
.pe-answer p { margin: 0 0 10px; color: var(--ink); font: 400 1rem/1.55 var(--font-sans); }

/* Source record cards */
.pe-source { background: var(--surface); border-radius: 12px; padding: 14px 18px; margin: 0 0 10px; }
.pe-source .pe-src-head { display: flex; gap: 10px; align-items: baseline; font: 400 0.8125rem/1.4 var(--font-mono);
                          color: var(--ink-muted); margin-bottom: 6px; }
.pe-source .pe-src-num { color: var(--ember-deep); font-weight: 600; }
.pe-source blockquote { margin: 0; padding: 0 0 0 12px; border-left: 2px solid var(--rule);
                        font: italic 400 1.0625rem/1.45 var(--font-serif); color: var(--ink); }
.pe-pointers { background: var(--panel); border-radius: 12px; padding: 14px 18px; margin-top: 8px; }
.pe-pointers p { margin: 0 0 6px; font: 400 0.875rem/1.4 var(--font-sans); color: var(--ink-muted); }

/* Stat tiles */
.pe-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 16px 0 24px; }
.pe-stat { background: var(--surface); border-radius: 12px; padding: 16px 18px; }
.pe-stat .v { font: 400 2.25rem/1 var(--font-serif); letter-spacing: -0.02em; color: var(--ink);
              font-variant-numeric: tabular-nums; }
.pe-stat .l { font: 500 0.75rem/1.3 var(--font-sans); letter-spacing: 0.08em; text-transform: uppercase;
              color: var(--ink-muted); margin-top: 8px; }
@media (max-width: 900px) { .pe-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

/* Pastel card trio */
.pe-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 24px; margin: 24px 0; }
.pe-card { border-radius: 24px; padding: 28px; color: var(--ink); }
.pe-card--sage { background: var(--sage); } .pe-card--lilac { background: var(--lilac); }
.pe-card--peach { background: var(--peach); }
.pe-card h4 { font: 500 1.375rem/1.1 var(--font-serif) !important; letter-spacing: -0.02em; margin: 0 0 8px; }
.pe-card p { font: 400 0.95rem/1.45 var(--font-sans); color: var(--on-pastel-body); margin: 0 0 10px; }
.pe-card code { font: 400 0.8125rem/1.4 var(--font-mono); background: transparent; color: var(--on-pastel-body);
                padding: 0; }
@media (max-width: 900px) { .pe-cards { grid-template-columns: 1fr; } }

[data-testid="stChatMessage"] { border-radius: 16px; padding: 16px 20px; }
[data-testid="stChatMessage"]:has(.pe-pill) { background: var(--surface); }
[data-testid="stTabs"] [role="tab"] p { font-family: var(--font-sans); font-weight: 500; }
.pe-footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
             border-top: 1px solid var(--rule);
             margin-top: 56px; padding-top: 16px; font: 400 0.875rem/1.35 var(--font-sans); color: var(--ink-muted); }
</style>
"""

LABEL_TEXT = {
    "grounded": "Grounded — every statement verified",
    "partially_grounded": "Partly grounded — some statements removed",
    "not_grounded": "Withheld — citations could not be verified",
    "declined": "Not in the indexed guidance",
    "blocked": "Blocked by the input guard",
}


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, ember_line: str, subtitle: str) -> None:
    st.markdown(
        f'<p class="pe-eyebrow">{escape(kicker)}</p>'
        f'<h1 class="pe-title">{escape(title)}<br><span class="pe-ember">{escape(ember_line)}</span></h1>'
        f'<p class="pe-sub">{escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def section_heading(kicker: str, title: str, subtitle: str = "") -> None:
    sub = f'<p class="pe-sub">{escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<p class="pe-eyebrow">{escape(kicker)}</p><h2 class="pe-heading">{escape(title)}</h2>{sub}',
        unsafe_allow_html=True,
    )


def status_row(label: str, chips: list[str]) -> None:
    pill = f'<span class="pe-pill pe-pill--{escape(label)}">{escape(LABEL_TEXT.get(label, label))}</span>'
    chip_html = "".join(f'<span class="pe-chip">{escape(c)}</span>' for c in chips)
    st.markdown(f'<div class="pe-row">{pill}{chip_html}</div>', unsafe_allow_html=True)


def answer_body(statements: list[dict[str, object]]) -> None:
    paragraphs = []
    for statement in statements:
        numbers = dict.fromkeys(int(n) for n in statement["sources"])  # type: ignore[attr-defined]
        marks = "".join(f"<sup>[{n}]</sup>" for n in numbers)  # each source once, in order
        paragraphs.append(f"<p>{escape(str(statement['text']))}{marks}</p>")
    st.markdown(f'<div class="pe-answer">{"".join(paragraphs)}</div>', unsafe_allow_html=True)


def source_card(number: int, label: str, quote: str) -> None:
    st.markdown(
        f'<div class="pe-source"><div class="pe-src-head"><span class="pe-src-num">[{number}]</span>'
        f"<span>{escape(label)}</span></div><blockquote>{escape(quote)}</blockquote></div>",
        unsafe_allow_html=True,
    )


def pointers(items: list[dict[str, str]]) -> None:
    links = "".join(f'<p>· <a href="{escape(p["url"], quote=True)}" target="_blank" rel="noopener">'
                    f"{escape(p['title'])}</a></p>" for p in items)  # fmt: skip
    st.markdown(
        f'<div class="pe-pointers"><p class="pe-eyebrow">Pages to check · web search, not verified</p>{links}</div>',
        unsafe_allow_html=True,
    )


def stat_tiles(tiles: list[tuple[str, str]]) -> None:
    html = "".join(f'<div class="pe-stat"><div class="v">{escape(v)}</div><div class="l">{escape(lbl)}</div></div>'
                   for v, lbl in tiles)  # fmt: skip
    st.markdown(f'<div class="pe-stats">{html}</div>', unsafe_allow_html=True)


def card_trio(cards: list[tuple[str, str, str]]) -> None:
    """Three pastel cards (sage → lilac → peach): (title, body, mono line)."""
    html = "".join(
        f'<div class="pe-card pe-card--{colour}"><h4>{escape(title)}</h4><p>{escape(body)}</p>'
        f"<code>{escape(mono)}</code></div>"
        for colour, (title, body, mono) in zip(["sage", "lilac", "peach"], cards, strict=False)
    )
    st.markdown(f'<div class="pe-cards">{html}</div>', unsafe_allow_html=True)


def footer(left: str, right: str) -> None:
    st.markdown(
        f'<div class="pe-footer"><span>{escape(left)}</span><span>{escape(right)}</span></div>',
        unsafe_allow_html=True,
    )
