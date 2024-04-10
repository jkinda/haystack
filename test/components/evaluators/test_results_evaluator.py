from pandas import DataFrame

from haystack.components.evaluators.results_evaluator import EvaluationResults


def test_init_results_evaluator():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    _ = EvaluationResults(pipeline_name="testing_pipeline_1", results=data)


def test_score_report():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    evaluator = EvaluationResults(pipeline_name="testing_pipeline_1", results=data)
    assert evaluator.to_pandas() == {
        "reciprocal_rank": 0.476932,
        "single_hit": 0.75,
        "multi_hit": 0.46428375,
        "context_relevance": 0.5817797499999999,
        "faithfulness": 0.40585374999999996,
        "semantic_answer_similarity": 0.53757075,
    }


def test_to_pandas():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7", "53c3b3e6", "225f87f7"],
            "question": [
                "What is the capital of France?",
                "What is the capital of Spain?",
                "What is the capital of Luxembourg?",
                "What is the capital of Portugal?",
            ],
            "contexts": ["wiki_France", "wiki_Spain", "wiki_Luxembourg", "wiki_Portugal"],
            "answer": ["Paris", "Madrid", "Luxembourg", "Lisbon"],
            "predicted_answer": ["Paris", "Madrid", "Luxembourg", "Lisbon"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    evaluator = EvaluationResults(pipeline_name="testing_pipeline_1", results=data)
    assert evaluator.to_pandas().equals(
        DataFrame(
            {
                "reciprocal_rank": [[0.378064, 0.534964, 0.216058, 0.778642]],
                "single_hit": [[1, 1, 0, 1]],
                "multi_hit": [[0.706125, 0.454976, 0.445512, 0.250522]],
                "context_relevance": [[0.805466, 0.410251, 0.750070, 0.361332]],
                "faithfulness": [[0.135581, 0.695974, 0.749861, 0.041999]],
                "semantic_answer_similarity": [[0.971241, 0.159320, 0.019722, 1]],
            }
        )
    )


def test_comparative_detailed_score_report():
    data_1 = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    data_2 = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    evaluator_1 = EvaluationResults(pipeline_name="testing_pipeline_1", results=data_1)
    evaluator_2 = EvaluationResults(pipeline_name="testing_pipeline_2", results=data_2)
    results = evaluator_1.comparative_individual_score_report(evaluator_2)
    print(results)
