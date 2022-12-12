import logging

import pandas as pd
import pytest

from haystack.schema import Document, Answer
from haystack.pipelines.base import Pipeline


@pytest.fixture
def table1():
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": ["58", "47", "60"],
        "number of movies": ["87", "53", "69"],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    table = pd.DataFrame(data)
    return table


@pytest.fixture
def table2():
    data = {
        "actors": ["chris pratt", "gal gadot", "oprah winfrey"],
        "age": ["45", "36", "65"],
        "number of movies": ["49", "34", "5"],
        "date of birth": ["12 january 1975", "5 april 1980", "15 september 1960"],
    }
    table = pd.DataFrame(data)
    return table


@pytest.mark.parametrize("table_reader_and_param", ["tapas_small", "tapas_scored"], indirect=True)
def test_table_reader(table_reader_and_param, table1, table2):
    table_reader, param = table_reader_and_param
    query = "When was Di Caprio born?"
    prediction = table_reader.predict(
        query=query,
        documents=[Document(content=table1, content_type="table"), Document(content=table2, content_type="table")],
    )

    assert prediction["query"] == "When was Di Caprio born?"

    # Check number of answers
    reference0 = {"tapas_small": {"num_answers": 2}, "rci": {"num_answers": 6}, "tapas_scored": {"num_answers": 6}}
    assert len(prediction["answers"]) == reference0[param]["num_answers"]

    # Check the first answer in the list
    reference1 = {"tapas_small": {"score": 1.0}, "rci": {"score": -6.5301}, "tapas_scored": {"score": 0.50568}}
    assert prediction["answers"][0].score == pytest.approx(reference1[param]["score"], rel=1e-3)
    assert prediction["answers"][0].answer == "11 november 1974"
    assert prediction["answers"][0].offsets_in_context[0].start == 7
    assert prediction["answers"][0].offsets_in_context[0].end == 8

    # Check the second answer in the list
    reference2 = {
        "tapas_small": {"answer": "5 april 1980", "start": 7, "end": 8, "score": 0.86314},
        "rci": {"answer": "47", "start": 5, "end": 6, "score": -6.836},
        "tapas_scored": {"answer": "brad pitt", "start": 0, "end": 1, "score": 0.49078},
    }
    assert prediction["answers"][1].score == pytest.approx(reference2[param]["score"], rel=1e-3)
    assert prediction["answers"][1].answer == reference2[param]["answer"]
    assert prediction["answers"][1].offsets_in_context[0].start == reference2[param]["start"]
    assert prediction["answers"][1].offsets_in_context[0].end == reference2[param]["end"]


@pytest.mark.parametrize("table_reader_and_param", ["tapas_small", "tapas_scored"], indirect=True)
def test_table_reader_batch_single_query_single_doc_list(table_reader_and_param, table1, table2):
    table_reader, param = table_reader_and_param
    query = "When was Di Caprio born?"
    prediction = table_reader.predict_batch(
        queries=[query],
        documents=[Document(content=table1, content_type="table"), Document(content=table2, content_type="table")],
    )
    # Expected output: List of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert prediction["queries"] == ["When was Di Caprio born?", "When was Di Caprio born?"]
    assert len(prediction["answers"]) == 2

    # Check number of answers for each document
    reference0 = {
        "tapas_small": {"num_answers": [1, 1]},
        "rci": {"num_answers": [3, 3]},
        "tapas_scored": {"num_answers": [3, 3]},
    }
    for i, ans_list in enumerate(prediction["answers"]):
        assert len(ans_list) == reference0[param]["num_answers"][i]

    # Check first answer from the 1ST document
    reference1 = {"tapas_small": {"score": 1.0}, "rci": {"score": -6.5301}, "tapas_scored": {"score": 0.50568}}
    assert prediction["answers"][0][0].score == pytest.approx(reference1[param]["score"], rel=1e-3)
    assert prediction["answers"][0][0].answer == "11 november 1974"
    assert prediction["answers"][0][0].offsets_in_context[0].start == 7
    assert prediction["answers"][0][0].offsets_in_context[0].end == 8

    # Check first answer from the 2ND Document
    reference2 = {
        "tapas_small": {"answer": "5 april 1980", "start": 7, "end": 8, "score": 0.86314},
        "rci": {"answer": "47", "start": 5, "end": 6, "score": -6.836},  # TODO Update once have RCI available
        "tapas_scored": {"answer": "5", "start": 10, "end": 11, "score": 0.11485},
    }
    assert prediction["answers"][1][0].score == pytest.approx(reference2[param]["score"], rel=1e-3)
    assert prediction["answers"][1][0].answer == reference2[param]["answer"]
    assert prediction["answers"][1][0].offsets_in_context[0].start == reference2[param]["start"]
    assert prediction["answers"][1][0].offsets_in_context[0].end == reference2[param]["end"]


@pytest.mark.parametrize("table_reader_and_param", ["tapas_small", "tapas_scored"], indirect=True)
def test_table_reader_batch_single_query_multiple_doc_lists(table_reader_and_param, table1):
    table_reader, param = table_reader_and_param
    query = "When was Di Caprio born?"
    prediction = table_reader.predict_batch(
        queries=[query], documents=[[Document(content=table1, content_type="table")]]
    )
    # Expected output: List of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 1


@pytest.mark.parametrize("table_reader_and_param", ["tapas_small", "tapas_scored"], indirect=True)
def test_table_reader_batch_multiple_queries_single_doc_list(table_reader_and_param, table1):
    table_reader, param = table_reader_and_param
    query = "When was Di Caprio born?"
    prediction = table_reader.predict_batch(
        queries=[query, query], documents=[Document(content=table1, content_type="table")]
    )
    # Expected output: List of lists of lists of answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], list)
    assert isinstance(prediction["answers"][0][0][0], Answer)
    assert len(prediction["answers"]) == 2  # Predictions for 2 queries


@pytest.mark.parametrize("table_reader_and_param", ["tapas_small", "tapas_scored"], indirect=True)
def test_table_reader_batch_multiple_queries_multiple_doc_lists(table_reader_and_param, table1):
    table_reader, param = table_reader_and_param
    query = "When was Di Caprio born?"
    prediction = table_reader.predict_batch(
        queries=[query, query],
        documents=[[Document(content=table1, content_type="table")], [Document(content=table1, content_type="table")]],
    )
    # Expected output: List of lists answers
    assert isinstance(prediction["answers"], list)
    assert isinstance(prediction["answers"][0], list)
    assert isinstance(prediction["answers"][0][0], Answer)
    assert len(prediction["answers"]) == 2  # Predictions for 2 collections of documents


@pytest.mark.parametrize("table_reader_and_param", ["tapas_small", "tapas_scored"], indirect=True)
def test_table_reader_in_pipeline(table_reader_and_param, table1):
    table_reader, param = table_reader_and_param
    pipeline = Pipeline()
    pipeline.add_node(table_reader, "TableReader", ["Query"])
    query = "When was Di Caprio born?"

    prediction = pipeline.run(query=query, documents=[Document(content=table1, content_type="table")])
    assert prediction["answers"][0].answer == "11 november 1974"
    assert prediction["answers"][0].offsets_in_context[0].start == 7
    assert prediction["answers"][0].offsets_in_context[0].end == 8


@pytest.mark.parametrize("table_reader_and_param", ["tapas_base"], indirect=True)
def test_table_reader_aggregation(table_reader_and_param):
    table_reader, param = table_reader_and_param
    data = {
        "Mountain": ["Mount Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu"],
        "Height": ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"],
    }
    table = pd.DataFrame(data)

    query = "How tall are all mountains on average?"
    prediction = table_reader.predict(query=query, documents=[Document(content=table, content_type="table")])
    assert prediction["answers"][0].score == pytest.approx(1.0)
    assert prediction["answers"][0].answer == "8609.2 m"
    assert prediction["answers"][0].meta["aggregation_operator"] == "AVERAGE"
    assert prediction["answers"][0].meta["answer_cells"] == ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"]

    query = "How tall are all mountains together?"
    prediction = table_reader.predict(query=query, documents=[Document(content=table, content_type="table")])
    assert prediction["answers"][0].score == pytest.approx(1.0)
    assert prediction["answers"][0].answer == "43046.0 m"
    assert prediction["answers"][0].meta["aggregation_operator"] == "SUM"
    assert prediction["answers"][0].meta["answer_cells"] == ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"]


@pytest.mark.parametrize("table_reader_and_param", ["tapas_small", "tapas_scored"], indirect=True)
def test_table_without_rows(caplog, table_reader_and_param):
    table_reader, param = table_reader_and_param
    # empty DataFrame
    table = pd.DataFrame()
    document = Document(content=table, content_type="table", id="no_rows")
    with caplog.at_level(logging.WARNING):
        predictions = table_reader.predict(query="test", documents=[document])
        assert "Skipping document with id 'no_rows'" in caplog.text
        assert len(predictions["answers"]) == 0


@pytest.mark.parametrize("table_reader_and_param", ["tapas_small", "tapas_scored"], indirect=True)
def test_text_document(caplog, table_reader_and_param):
    table_reader, param = table_reader_and_param
    document = Document(content="text", id="text_doc")
    with caplog.at_level(logging.WARNING):
        predictions = table_reader.predict(query="test", documents=[document])
        assert "Skipping document with id 'text_doc'" in caplog.text
        assert len(predictions["answers"]) == 0
