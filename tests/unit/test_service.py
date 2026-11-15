from regintel.answering.schema import AnswerResult, Usage, WebPointer
from regintel.ingestion.chunking import Chunk
from regintel.retrieval.base import Hit
from regintel.service import on_regulator_domain, web_pointers


class FakeWeb:
    name = "web"

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        pages = [
            ("https://www.ema.europa.eu/en/documents/annex-11-concept-paper", "Concept paper on Annex 11"),
            ("https://evil.example.com/ema.europa.eu/fake", "Lookalike"),
            ("https://health.ec.europa.eu/consultation", "Consultation"),
            ("https://www.fda.gov.attacker.net/page", "Suffix trick"),
        ]
        return [
            Hit(Chunk(f"web@{i}", f"web:{u}", "", "web", t, 0, 0, "summary", ""), 1.0) for i, (u, t) in enumerate(pages)
        ]


def test_only_regulator_domains_pass() -> None:
    assert on_regulator_domain("https://www.fda.gov/regulatory-information")
    assert on_regulator_domain("https://health.ec.europa.eu/x")
    assert not on_regulator_domain("https://www.fda.gov.attacker.net/page")
    assert not on_regulator_domain("https://evil.example.com/ema.europa.eu/fake")
    pointers = web_pointers(FakeWeb(), "q")  # type: ignore[arg-type]
    assert [p.title for p in pointers] == ["Concept paper on Annex 11", "Consultation"]


def test_pointers_follow_a_decline_and_are_marked_unverified() -> None:
    result = AnswerResult("q", "declined", [], "The indexed guidance does not address this.", [], Usage(), 0, 0)
    result.web_pointers = [WebPointer("Concept paper", "https://www.ema.europa.eu/x", "summary")]
    assert result.text.startswith("The indexed guidance does not address this.")
    assert "not verified" in result.text and "- Concept paper: https://www.ema.europa.eu/x" in result.text
