"""Generation-quality metrics: answer correctness, citation correctness,
groundedness, hallucination rate, correct refusal rate.

None of these use an LLM judge — the project has no reliable OpenAI access,
so every metric here is a
deliberately simple, structurally-checkable proxy rather than a semantic
judgment. docs/EVALUATION.md's Limitations section names this explicitly:
these numbers describe whether the pipeline *followed its contract*
(cited something, cited the right document, refused when it should have),
not whether individual sentences are factually correct — that would need an
LLM-as-judge pass, which is out of scope until OpenAI credits are available.
"""

from __future__ import annotations


def answer_correctness(answer: str, expected_answer_contains: list[str]) -> bool:
    """True if the answer contains at least one expected phrase (case-insensitive
    substring match). Entries in expected_answer_contains are alternative
    acceptable phrasings (OR, not AND) — see golden_qa.yaml's schema comment.
    Only meaningful for ANSWERABLE questions; pass expected_answer_contains=[]
    and this always returns False, which is correct for unanswerable ones too
    (nothing an answer could "correctly" contain) — but use
    correct_refusal_rate for that case instead, it's more informative.
    """
    if not expected_answer_contains:
        return False
    lowered = answer.lower()
    return any(phrase.lower() in lowered for phrase in expected_answer_contains)


def citation_correctness(cited_documents: set[str], expected_source_documents: list[str]) -> bool:
    """True if at least one citation points to an expected source document."""
    if not expected_source_documents:
        return False
    return bool(cited_documents & set(expected_source_documents))


def refused(answer: str, no_answer_text: str) -> bool:
    """True if the answer is (or reduces to) the pipeline's fixed refusal text."""
    return answer.strip() == no_answer_text.strip()


def has_citation_when_answering(
    answer: str, cited_chunk_indices: list[int], no_answer_text: str
) -> bool:
    """Structural groundedness proxy: the system prompt demands citing every
    claim, so a non-refusal answer with zero citations is treated as
    ungrounded. Cannot verify a cited chunk actually *supports* the claim
    next to it — only that the model followed the citation contract at all.
    """
    if refused(answer, no_answer_text):
        return True  # a refusal makes no claims, so there's nothing to ground
    return len(cited_chunk_indices) > 0


def groundedness_rate(
    answers_with_citations: list[tuple[str, list[int]]], no_answer_text: str
) -> float:
    """Fraction of ANSWERABLE-question answers that respect the citation
    contract (see has_citation_when_answering). Pass (answer, cited_chunk_indices)
    pairs for answerable golden questions only.
    """
    if not answers_with_citations:
        return 0.0
    grounded = sum(
        1
        for answer, indices in answers_with_citations
        if has_citation_when_answering(answer, indices, no_answer_text)
    )
    return grounded / len(answers_with_citations)


def correct_refusal_rate(unanswerable_answers: list[str], no_answer_text: str) -> float:
    """Fraction of UNANSWERABLE golden questions where the model correctly
    refused instead of fabricating an answer. Pass answers for questions
    where the golden dataset marks answerable=False only.
    """
    if not unanswerable_answers:
        return 0.0
    correct = sum(1 for answer in unanswerable_answers if refused(answer, no_answer_text))
    return correct / len(unanswerable_answers)


def hallucination_rate(unanswerable_answers: list[str], no_answer_text: str) -> float:
    """Structural hallucination proxy on the UNANSWERABLE subset: the
    complement of correct_refusal_rate — the model answered instead of
    refusing when the corpus had nothing to support an answer. Cannot detect
    a subtly wrong fact embedded in an otherwise-plausible answer to an
    ANSWERABLE question; that needs an LLM judge (see module docstring).
    """
    return 1.0 - correct_refusal_rate(unanswerable_answers, no_answer_text)
