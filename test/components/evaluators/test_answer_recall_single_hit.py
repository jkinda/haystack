import pytest

from haystack.components.evaluators.answer_recall_single_hit import AnswerRecallSingleHitEvaluator


def test_run_with_all_matching():
    evaluator = AnswerRecallSingleHitEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["Paris"]],
    )

    assert result == {"scores": [1.0, 1.0], "average": 1.0}


def test_run_with_no_matching():
    evaluator = AnswerRecallSingleHitEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Paris"], ["London"]],
    )

    assert result == {"scores": [0.0, 0.0], "average": 0.0}


def test_run_with_partial_matching():
    evaluator = AnswerRecallSingleHitEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["London"]],
    )

    assert result == {"scores": [1.0, 0.0], "average": 0.5}


def test_run_with_complex_data():
    evaluator = AnswerRecallSingleHitEvaluator()
    result = evaluator.run(
        questions=[
            "In what country is Normandy located?",
            "When was the Latin version of the word Norman first recorded?",
            "What developed in Normandy during the 1100s?",
            "In what century did important classical music developments occur in Normandy?",
            "From which countries did the Norse originate?",
            "What century did the Normans first gain their separate identity?",
        ],
        ground_truth_answers=[
            ["France"],
            ["9th century", "9th"],
            ["classical music", "classical"],
            ["11th century", "the 11th"],
            ["Denmark", "Iceland", "Norway", "Denmark, Iceland and Norway"],
            ["10th century", "10th"],
        ],
        predicted_answers=[
            ["France"],
            ["9th century", "10th century", "9th"],
            ["classical", "rock music", "dubstep"],
            ["11th", "the 11th", "11th century"],
            ["Denmark", "Norway", "Iceland"],
            ["10th century", "the first half of the 10th century", "10th", "10th"],
        ],
    )
    assert result == {"scores": [1.0, 1.0, 0.5, 1.0, 0.75, 1.0], "average": 0.875}


def test_run_with_different_lengths():
    evaluator = AnswerRecallSingleHitEvaluator()
    with pytest.raises(ValueError):
        evaluator.run(
            questions=["What is the capital of Germany?"],
            ground_truth_answers=[["Berlin"], ["Paris"]],
            predicted_answers=[["Berlin"], ["London"]],
        )

    with pytest.raises(ValueError):
        evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_answers=[["Berlin"]],
            predicted_answers=[["Berlin"], ["London"]],
        )

    with pytest.raises(ValueError):
        evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_answers=[["Berlin"], ["Paris"]],
            predicted_answers=[["Berlin"]],
        )
