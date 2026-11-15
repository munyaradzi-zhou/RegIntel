"""FAERS charts (Altair), following the dataviz skill's mark specs.

- Reports per reaction: magnitude across categories -> horizontal bars, one series, one hue
  (violet), no legend, value labels on the bars, recessive grid, hover tooltip.
- Reporting odds ratio: an estimate with an interval -> a forest plot on a log scale, a dot for
  the ROR and a line for its 95% CI, with a reference line at 1 (no disproportion). Reactions are
  in the same order as the bar chart so the two read together.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

from regintel.frontend.theme import INK, INK_MUTED, MARK, RULE

FONT = "Inter, Helvetica Neue, Arial, sans-serif"


def _style(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    styled: alt.Chart | alt.LayerChart = (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=INK_MUTED,
            titleColor=INK_MUTED,
            gridColor=RULE,
            domainColor=RULE,
            tickColor=RULE,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight=500,
        )
        .configure(background="transparent")
    )
    return styled


def reactions_bar(profile: dict[str, Any]) -> alt.Chart | alt.LayerChart:
    data = pd.DataFrame(profile["top_reactions"])
    order = list(data["reaction"])
    base = alt.Chart(data).encode(
        y=alt.Y("reaction:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, ticks=False, domain=False)),
        x=alt.X("reports:Q", title="Reports listing the drug as suspect", axis=alt.Axis(tickCount=4, grid=True)),
        tooltip=[alt.Tooltip("reaction:N", title="Reaction"), alt.Tooltip("reports:Q", title="Reports", format=",")],
    )
    bars = base.mark_bar(color=MARK, cornerRadiusEnd=4, height={"band": 0.62})
    labels = base.mark_text(align="left", dx=6, font=FONT, fontSize=12, color=INK).encode(
        text=alt.Text("reports:Q", format=",")
    )
    return _style((bars + labels).properties(height=34 * len(data) + 20))


def ror_forest(profile: dict[str, Any]) -> alt.Chart | alt.LayerChart:
    rows = [
        {"reaction": s["reaction"], "ror": s["ror"], "low": s["ci95"][0], "high": s["ci95"][1], "reports": s["a"]}
        for s in profile["signals"]
        if s["ror"] is not None and s["ci95"]
    ]
    data = pd.DataFrame(rows)
    order = [r["reaction"] for r in profile["top_reactions"]]
    lo = min(0.25, float(data["low"].min()) * 0.8) if len(data) else 0.25
    hi = max(4.0, float(data["high"].max()) * 1.25) if len(data) else 4.0
    ticks = [v for v in (0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64) if lo <= v <= hi]  # doublings only
    x = alt.X(
        "ror:Q",
        title="Reporting odds ratio (log scale, 95% CI)",
        scale=alt.Scale(type="log", domain=[lo, hi]),
        axis=alt.Axis(grid=True, values=ticks, format="~g"),
    )
    y = alt.Y("reaction:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, ticks=False, domain=False))
    tooltip = [
        alt.Tooltip("reaction:N", title="Reaction"),
        alt.Tooltip("ror:Q", title="ROR", format=".2f"),
        alt.Tooltip("low:Q", title="95% CI from", format=".2f"),
        alt.Tooltip("high:Q", title="to", format=".2f"),
        alt.Tooltip("reports:Q", title="Reports with both", format=","),
    ]
    reference = alt.Chart(pd.DataFrame({"ror": [1.0]})).mark_rule(color=INK_MUTED, strokeDash=[4, 4]).encode(x="ror:Q")
    interval = (
        alt.Chart(data)
        .mark_rule(color=MARK, strokeWidth=2)
        .encode(x=alt.X("low:Q", scale=alt.Scale(type="log", domain=[lo, hi])), x2="high:Q", y=y, tooltip=tooltip)
    )
    dots = (
        alt.Chart(data)
        .mark_point(filled=True, size=90, color=MARK, stroke="#FAF4E8", strokeWidth=2)
        .encode(x=x, y=y, tooltip=tooltip)
    )
    return _style((reference + interval + dots).properties(height=34 * max(1, len(data)) + 20))
