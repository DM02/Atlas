from eval.metrics.generation import (
    answer_correctness,
    citation_correctness,
    correct_refusal_rate,
    groundedness_rate,
    hallucination_rate,
    has_citation_when_answering,
    refused,
)

NO_ANSWER_TEXT = "I don't have enough information in the provided documents to answer that."


def test_answer_correctness_matches_one_of_alternative_phrasings() -> None:
    assert answer_correctness("Employees get three days remote per week.", ["three days", "3 days"])
    assert answer_correctness("You get 3 days of remote work.", ["three days", "3 days"])


def test_answer_correctness_case_insensitive() -> None:
    assert answer_correctness("THE ANSWER IS 15 DAYS", ["15 days"])


def test_answer_correctness_false_when_phrase_absent() -> None:
    phrases = ["three days", "3 days"]
    assert not answer_correctness("Employees get two days remote per week.", phrases)


def test_answer_correctness_false_with_no_expected_phrases() -> None:
    assert not answer_correctness("anything", [])


def test_citation_correctness_true_on_overlap() -> None:
    assert citation_correctness({"employee_handbook.txt"}, ["employee_handbook.txt"])


def test_citation_correctness_false_without_overlap() -> None:
    assert not citation_correctness({"product_overview.txt"}, ["employee_handbook.txt"])


def test_citation_correctness_false_when_nothing_expected() -> None:
    assert not citation_correctness({"employee_handbook.txt"}, [])


def test_refused_matches_exact_no_answer_text() -> None:
    assert refused(NO_ANSWER_TEXT, NO_ANSWER_TEXT)
    assert not refused("15 days [1]", NO_ANSWER_TEXT)


def test_has_citation_when_answering_true_for_refusal() -> None:
    assert has_citation_when_answering(NO_ANSWER_TEXT, [], NO_ANSWER_TEXT)


def test_has_citation_when_answering_false_for_uncited_claim() -> None:
    assert not has_citation_when_answering("15 days.", [], NO_ANSWER_TEXT)


def test_has_citation_when_answering_true_when_cited() -> None:
    assert has_citation_when_answering("15 days [1].", [1], NO_ANSWER_TEXT)


def test_groundedness_rate_mixed() -> None:
    pairs = [
        ("15 days [1].", [1]),
        ("20 days.", []),  # ungrounded: claim with no citation
        (NO_ANSWER_TEXT, []),  # refusal: trivially grounded
    ]

    assert groundedness_rate(pairs, NO_ANSWER_TEXT) == 2 / 3


def test_groundedness_rate_empty_is_zero() -> None:
    assert groundedness_rate([], NO_ANSWER_TEXT) == 0.0


def test_correct_refusal_rate_all_refused() -> None:
    assert correct_refusal_rate([NO_ANSWER_TEXT, NO_ANSWER_TEXT], NO_ANSWER_TEXT) == 1.0


def test_correct_refusal_rate_none_refused() -> None:
    assert correct_refusal_rate(["some fabricated answer"], NO_ANSWER_TEXT) == 0.0


def test_correct_refusal_rate_empty_is_zero() -> None:
    assert correct_refusal_rate([], NO_ANSWER_TEXT) == 0.0


def test_hallucination_rate_is_complement_of_refusal_rate() -> None:
    answers = [NO_ANSWER_TEXT, "fabricated answer"]

    assert hallucination_rate(answers, NO_ANSWER_TEXT) == 0.5
