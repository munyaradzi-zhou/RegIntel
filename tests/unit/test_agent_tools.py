from regintel.agent.evidence import EvidencePool
from regintel.agent.tools import ResearchTools
from regintel.answering.pipeline import verify_draft
from regintel.answering.schema import DraftAnswer, DraftCitation, DraftStatement
from regintel.ingestion.chunking import Chunk
from regintel.retrieval.base import Hit
from regintel.terminology.cdisc import Term, Terminology

E2A = Chunk(
    "ich-e2a#0010", "ich-e2a", "ICH", "III.B.1", "Fatal ADRs", 7, 7, "no later than 7 calendar days", "[ICH E2A]"
)
OTHER = Chunk(
    "ich-e2a#0011", "ich-e2a", "ICH", "III.B.2", "Other ADRs", 7, 7, "no later than 15 calendar days", "[ICH E2A]"
)

SEVERITY = Term(
    "sdtm", "C66769", "", "Severity/Intensity Scale for Adverse Events", "AESEV", (), "A scale.", "Severity"
)
MILD = Term(
    "sdtm", "C41338", "C66769", "Severity/Intensity Scale for Adverse Events", "MILD", ("Mild",), "Mild.", "Mild"
)


class FakeRetriever:
    name = "fake"

    def __init__(self, *hits: Chunk) -> None:
        self.hits = [Hit(c, 1.0) for c in hits]
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        self.queries.append(query)
        return self.hits[:top_k]


def test_pool_numbers_in_arrival_order_and_dedupes_passages() -> None:
    pool = EvidencePool()
    assert pool.add("guidance", E2A).number == 1
    assert pool.add("guidance", OTHER).number == 2
    assert pool.add("guidance", E2A).number == 1  # found again by a second search
    record = pool.add_record("faers", "faers:summary:x", "FAERS extract · computed", "Reports: 1,234.")
    assert record.number == 3 and record.chunk.page_start == 0
    assert [c.chunk_id for c in pool.chunks()] == ["ich-e2a#0010", "ich-e2a#0011", "faers:summary:x"]


def test_search_tool_records_evidence_and_trace() -> None:
    tools = ResearchTools(FakeRetriever(E2A, OTHER))
    rendered = tools.search_guidance("E2A fatal ADR timeline")
    assert rendered.startswith("[E1] [ICH E2A]") and "[E2]" in rendered
    assert tools.calls[0].name == "search_guidance" and tools.calls[0].evidence == [1, 2]


def test_cdisc_tool_describes_codelists_with_members() -> None:
    tools = ResearchTools(FakeRetriever(), terminology=Terminology([SEVERITY, MILD]), terminology_version="test")
    text = tools.lookup_cdisc_term("AESEV")
    assert "Codelist Severity/Intensity Scale for Adverse Events (C66769), submission value AESEV" in text
    assert "It has 1 terms: MILD (C41338)" in text
    assert "No CDISC Controlled Terminology entry matches 'XYZ'" in tools.lookup_cdisc_term("XYZ")


def test_unavailable_tools_are_not_offered() -> None:
    names = [t.name for t in ResearchTools(FakeRetriever()).as_tools()]
    assert names == ["search_guidance"]


def test_tool_evidence_is_verifiable_like_guidance() -> None:
    tools = ResearchTools(FakeRetriever(E2A))
    tools.search_guidance("q")
    tools.pool.add_record("faers", "faers:x", "FAERS extract · computed", "Reports listing X as a suspect drug: 1,936.")
    draft = DraftAnswer(
        answerable=True,
        statements=[
            DraftStatement(
                text="1,936 reports list X.",
                citations=[DraftCitation(passage=2, quote="Reports listing X as a suspect drug: 1,936")],
            ),
            DraftStatement(
                text="2,000 reports list X.",
                citations=[DraftCitation(passage=2, quote="Reports listing X as a suspect drug: 1,936")],
            ),
        ],
        decline_reason=None,
    )
    result = verify_draft("q", draft, tools.pool.chunks(), None)
    assert [s.verified for s in result.statements] == [True, False]  # a wrong count is caught
    assert result.sources[0].source_label == "FAERS extract · computed" and result.sources[0].pages == ""
