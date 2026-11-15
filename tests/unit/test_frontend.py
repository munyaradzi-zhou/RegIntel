"""Front-end helpers: SSE parsing, HTML escaping of API text, chart specifications."""

from typing import Any

import pytest

from regintel.frontend import charts, theme
from regintel.frontend.client import parse_sse

PROFILE: dict[str, Any] = {
    "top_reactions": [{"reaction": "Nausea", "reports": 36}, {"reaction": "Rash", "reports": 12}],
    "signals": [
        {"reaction": "Nausea", "ror": 6.23, "ci95": [4.31, 9.02], "a": 36},
        {"reaction": "Rash", "ror": None, "ci95": None, "a": 0},  # not computable: left out of the plot
    ],
}


def test_sse_events_split_on_blank_lines() -> None:
    lines = ["event: session", 'data: {"session_id": "s1"}', "", "event: progress", 'data: {"elapsed": 1.5}', "",
             'data: {"no": "event name"}']  # fmt: skip
    assert list(parse_sse(iter(lines))) == [
        ("session", {"session_id": "s1"}),
        ("progress", {"elapsed": 1.5}),
        ("message", {"no": "event name"}),
    ]


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    html: list[str] = []
    monkeypatch.setattr(theme.st, "markdown", lambda body, **_: html.append(body))
    return html


def test_api_text_is_escaped_before_it_reaches_html(rendered: list[str]) -> None:
    theme.answer_body([{"text": "<script>alert(1)</script> & co", "sources": [2, 1, 2]}])
    theme.source_card(1, "EMA <b>guideline</b>", 'a "quote" <img src=x>')
    theme.pointers([{"title": "<i>t</i>", "url": 'https://www.ema.europa.eu/x" onclick="y'}])
    joined = "".join(rendered)
    assert "<script>" not in joined and "&lt;script&gt;" in joined
    assert "<img" not in joined and "<b>guideline" not in joined and "<i>t</i>" not in joined
    assert '" onclick="y' not in joined  # attribute break-out is escaped
    assert "<sup>[2]</sup><sup>[1]</sup></p>" in joined  # each source once, in first-cited order


def test_status_pill_names_the_label_in_words(rendered: list[str]) -> None:
    theme.status_row("partially_grounded", ["path · guidance", "<x>"])
    assert "Partly grounded" in rendered[0] and "pe-pill--partially_grounded" in rendered[0]
    assert "&lt;x&gt;" in rendered[0]


def test_chart_specs_use_the_validated_mark_colour_and_a_log_scale() -> None:
    bars = charts.reactions_bar(PROFILE).to_dict()
    assert uses_mark_colour(bars) and bars["layer"][0]["mark"]["type"] == "bar"
    forest = charts.ror_forest(PROFILE).to_dict()
    assert forest["layer"][2]["encoding"]["x"]["scale"]["type"] == "log"
    assert [row["reaction"] for row in forest["datasets"][forest["layer"][2]["data"]["name"]]] == ["Nausea"]


def uses_mark_colour(spec: dict[str, Any]) -> bool:
    return theme.MARK in str(spec)
