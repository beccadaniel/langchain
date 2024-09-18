# Unit test class
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import sqlalchemy
from langchain_core.documents.base import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.sqlserver import (
    DistanceStrategy,
    SQLServer_VectorStore,
)

EMBEDDING_LENGTH = 1536


def generalized_mock_factory() -> Tuple[SQLServer_VectorStore, Dict[str, MagicMock]]:
    mocks = {
        "_create_engine": MagicMock(),
        "_prepare_json_data_type": MagicMock(),
        "_get_embedding_store": MagicMock(),
        "_create_table_if_not_exists": MagicMock(),
        "_can_connect_with_entra_id": MagicMock(),
        "_provide_token": MagicMock(return_value=True),
        "_handle_field_filter": MagicMock(),
        "_docs_from_result": MagicMock(),
        "_docs_and_scores_from_result": MagicMock(),
        "_insert_embeddings": MagicMock(),
        "delete": MagicMock(),
        "_delete_texts_by_ids": MagicMock(),
        "similarity_search": MagicMock(),
        "similarity_search_by_vector": MagicMock(),
        "similarity_search_with_score": MagicMock(),
        "similarity_search_by_vector_with_score": MagicMock(),
        "add_texts": MagicMock(),
        "drop": MagicMock(),
        "_create_filter_clause": MagicMock(),
        "_search_store": MagicMock(),
    }

    with ExitStack() as stack:
        for method, mock in mocks.items():
            stack.enter_context(
                patch(
                    f"langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.{method}",
                    mock,
                )
            )

        connection_string = "mssql://abcde.database.windows.net,1433/iamvectorstore?driver=ODBC+Driver+17+for+SQL+Server"
        db_schema = "test_schema"
        distance_strategy = DistanceStrategy.DOT
        embedding_function = FakeEmbeddings(size=128)
        embedding_length = 128
        table_name = "test_table"

        store = SQLServer_VectorStore(
            connection_string=connection_string,
            db_schema=db_schema,
            distance_strategy=distance_strategy,
            embedding_function=embedding_function,
            embedding_length=embedding_length,
            table_name=table_name,
        )

    return store, mocks


def test_init() -> None:
    # Arrange
    store, mocks = generalized_mock_factory()

    # Assert
    assert (
        store.connection_string
        == "mssql://abcde.database.windows.net,1433/iamvectorstore?driver=ODBC+Driver+17+for+SQL+Server"
    )
    assert store._distance_strategy == DistanceStrategy.DOT
    assert store.embedding_function == FakeEmbeddings(size=128)
    assert store._embedding_length == 128
    assert store.schema == "test_schema"
    assert store.table_name == "test_table"
    mocks["_create_engine"].assert_called_once()
    mocks["_prepare_json_data_type"].assert_called_once()
    mocks["_get_embedding_store"].assert_called_once_with("test_table", "test_schema")
    mocks["_create_table_if_not_exists"].assert_called_once()


def test_can_connect_with_entra_id() -> None:
    store, mocks = generalized_mock_factory()
    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._can_connect_with_entra_id",
        wraps=SQLServer_VectorStore._can_connect_with_entra_id,
    ), patch(
        "langchain_community.vectorstores.sqlserver.urlparse", wraps=MagicMock()
    ) as mock_urlparse:
        # case 1: parsed_url is None
        mock_urlparse.return_value = None
        result = store._can_connect_with_entra_id(store)
        mock_urlparse.assert_called_once_with(store.connection_string)
        assert result is False

        mock_urlparse.reset_mock()

        # case 2: parsed_url has username and password
        url_value = {
            "username": "username123",
            "password": "password123",
        }

        json_string = json.dumps(url_value, indent=4)

        parsed_json = json.loads(
            json_string, object_hook=lambda d: SimpleNamespace(**d)
        )

        mock_urlparse.return_value = parsed_json

        result = store._can_connect_with_entra_id(store)
        mock_urlparse.assert_called_once_with(store.connection_string)
        assert result is False

        mock_urlparse.reset_mock()

        # case 3: parsed_url has trusted_connection=yes
        url_value = {
            "username": "",
            "password": "",
            "query": "trusted_connection=yes",
        }

        json_string = json.dumps(url_value, indent=4)

        parsed_json = json.loads(
            json_string, object_hook=lambda d: SimpleNamespace(**d)
        )
        mock_urlparse.return_value = parsed_json

        result = store._can_connect_with_entra_id(store)
        mock_urlparse.assert_called_once_with(store.connection_string)
        assert result is False

        mock_urlparse.reset_mock()

        # case 4: parsed_url does not have trusted_connection=yes,
        #  no username and password
        url_value = {
            "username": "",
            "password": "",
            "query": "trusted_connection=no",
        }

        json_string = json.dumps(url_value, indent=4)

        parsed_json = json.loads(
            json_string, object_hook=lambda d: SimpleNamespace(**d)
        )
        mock_urlparse.return_value = parsed_json

        result = store._can_connect_with_entra_id(store)
        mock_urlparse.assert_called_once_with(store.connection_string)
        assert result is True


def test_similarity_search() -> None:
    store, mocks = generalized_mock_factory()

    query = "hi"
    mock_responses = {"hello": [0.1, 0.2, 0.3], "hi": [0.01, 0.02, 0.03]}

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search",
        wraps=SQLServer_VectorStore.similarity_search,
    ), patch.object(
        store, "similarity_search_by_vector", wraps=mocks["similarity_search_by_vector"]
    ):
        store.embedding_function = Mock()
        store.embedding_function.embed_query = Mock()
        store.embedding_function.embed_query.return_value = mock_responses[query]
        mocks["similarity_search_by_vector"].return_value = mock_responses

        store.similarity_search(store, query)

        mocks["similarity_search_by_vector"].assert_called_once_with(
            [0.01, 0.02, 0.03], 4
        )

        mocks["similarity_search_by_vector"].reset_mock()

        query = "hello"
        store.embedding_function.embed_query.reset_mock()
        store.embedding_function.embed_query.return_value = mock_responses[query]

        store.similarity_search(store, query, 7)

        mocks["similarity_search_by_vector"].assert_called_once_with([0.1, 0.2, 0.3], 7)


def test_similarity_search_by_vector() -> None:
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search_by_vector",
        wraps=SQLServer_VectorStore.similarity_search_by_vector,
    ), patch.object(
        store,
        "similarity_search_by_vector_with_score",
        wraps=mocks["similarity_search_by_vector_with_score"],
    ), patch.object(store, "_docs_from_result", wraps=mocks["_docs_from_result"]):
        embeddings = [0.1, 0.2, 0.3]
        mock_responses = {
            tuple(["0.01", "0.02", "0.03"]): (
                Document(
                    page_content="""Got these on sale for roughly 25 cents per cup"""
                ),
                0.9588668232580106,
            ),
        }
        expected_result = (
            Document(page_content="Got these on sale for roughly 25 cents per cup"),
            0.9588668232580106,
        )

        mocks["similarity_search_by_vector_with_score"].return_value = mock_responses[
            tuple(["0.01", "0.02", "0.03"])
        ]
        mocks["_docs_from_result"].return_value = expected_result

        store.similarity_search_by_vector(store, embeddings)

        mocks["similarity_search_by_vector_with_score"].assert_called_once_with(
            embeddings, 4
        )
        mocks["_docs_from_result"].assert_called_once_with(
            mock_responses[tuple(["0.01", "0.02", "0.03"])]
        )

        mocks["similarity_search_by_vector_with_score"].reset_mock()
        mocks["_docs_from_result"].reset_mock()

        store.similarity_search_by_vector(store, embeddings, 7)

        mocks["_docs_from_result"].assert_called_once_with(
            mock_responses[tuple(["0.01", "0.02", "0.03"])]
        )
        mocks["similarity_search_by_vector_with_score"].assert_called_once_with(
            [0.1, 0.2, 0.3], 7
        )
        mocks["_docs_from_result"].assert_called_once_with(
            mock_responses[tuple(["0.01", "0.02", "0.03"])]
        )


def test_similarity_search_wih_score() -> None:
    store, mocks = generalized_mock_factory()

    query = "hi"
    mock_responses = {"hello": [0.1, 0.2, 0.3], "hi": [0.01, 0.02, 0.03]}

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search_with_score",
        wraps=SQLServer_VectorStore.similarity_search_with_score,
    ), patch.object(
        store,
        "similarity_search_by_vector_with_score",
        wraps=mocks["similarity_search_by_vector_with_score"],
    ):
        store.embedding_function = Mock()
        store.embedding_function.embed_query = Mock()
        store.embedding_function.embed_query.return_value = mock_responses[query]
        mocks["similarity_search_by_vector_with_score"].return_value = mock_responses

        store.similarity_search_with_score(store, query)

        mocks["similarity_search_by_vector_with_score"].assert_called_once_with(
            [0.01, 0.02, 0.03], 4
        )

        mocks["similarity_search_by_vector_with_score"].reset_mock()

        query = "hello"
        store.embedding_function.embed_query.reset_mock()
        store.embedding_function.embed_query.return_value = mock_responses[query]

        store.similarity_search_with_score(store, query, 7)

        mocks["similarity_search_by_vector_with_score"].assert_called_once_with(
            [0.1, 0.2, 0.3], 7
        )


def test_similarity_search_by_vector_with_score() -> None:
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore."
        "similarity_search_by_vector_with_score",
        wraps=SQLServer_VectorStore.similarity_search_by_vector_with_score,
    ), patch.object(store, "_search_store", wraps=mocks["_search_store"]), patch.object(
        store,
        "_docs_and_scores_from_result",
        wraps=mocks["_docs_and_scores_from_result"],
    ):
        embeddings = tuple([0.01, 0.02, 0.03])

        expected_search_result = """(<langchain_community.vectorstores.sqlserver.
           SQLServer_VectorStore._get_embedding_store.<locals>.EmbeddingStore object
             at 0x0000025EFFF84810>,
                 0.9595672912317021)"""
        expected_docs = (
            Document(page_content="""Got these on sale for roughly 25 cents per cup"""),
            "0.9588668232580106",
        )

        mocks["_search_store"].return_value = expected_search_result
        mocks["_docs_and_scores_from_result"].return_value = expected_docs

        # case 1: k is not given
        result = store.similarity_search_by_vector_with_score(store, embeddings)

        mocks["_search_store"].assert_called_once_with(embeddings, 4)
        mocks["_docs_and_scores_from_result"].assert_called_once_with(
            expected_search_result
        )
        assert result == expected_docs

        mocks["_docs_and_scores_from_result"].reset_mock()
        mocks["_search_store"].reset_mock()

        # case 2: k =7
        result = store.similarity_search_by_vector_with_score(store, embeddings, 7)

        mocks["_search_store"].assert_called_once_with(embeddings, 7)
        mocks["_docs_and_scores_from_result"].assert_called_once_with(
            expected_search_result
        )
        assert result == expected_docs


def test_add_texts() -> None:
    store, mocks = generalized_mock_factory()

    texts = ["hi", "hello", "welcome"]
    embeddings = [0.01, 0.02, 0.03]
    metadatas = [
        {"id": 1, "summary": "Good Quality Dog Food"},
        {"id": 2, "summary": "Nasty No flavor"},
        {"id": 3, "summary": "stale product"},
    ]
    ids = [1, 2, 3]
    input_ids = [4, 5, 6]

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.add_texts",
        wraps=SQLServer_VectorStore.add_texts,
    ), patch.object(store, "_insert_embeddings", wraps=mocks["_insert_embeddings"]):
        store.embedding_function = Mock()
        store.embedding_function.embed_documents = Mock()
        store.embedding_function.embed_documents.return_value = embeddings
        mocks["_insert_embeddings"].return_value = ids

        # case 1:input ids not given
        returned_ids = store.add_texts(store, texts, metadatas)
        assert returned_ids == ids
        mocks["_insert_embeddings"].assert_called_once_with(
            texts, embeddings, metadatas, None
        )

        mocks["_insert_embeddings"].reset_mock()

        # case 1:input ids not given
        returned_ids = store.add_texts(store, texts, metadatas, input_ids)
        assert returned_ids == ids
        mocks["_insert_embeddings"].assert_called_once_with(
            texts, embeddings, metadatas, input_ids
        )


def test_create_filter_clause() -> None:
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_filter_clause",
        wraps=SQLServer_VectorStore._create_filter_clause,
    ), patch.object(
        store, "_handle_field_filter", wraps=mocks["_handle_field_filter"]
    ), patch("sqlalchemy.and_", wraps=MagicMock()) as mock_sqlalchemy_and:
        # filter case 0: Filters is not dict
        filter_value = ["hi"]

        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value)
        assert str(context.exception) == (
            """Expected a dict, but got <class 'list'> for value: <class 'filter'>"""
        )

        # filter case 1: Outer operator is not AND/OR
        filter_value_1 = {"$XOR": 2}

        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value_1)
        assert str(context.exception) == (
            """Invalid filter condition.\nExpected $and or $or but got: $XOR"""
        )

        # filter case 2: Valid field filter case
        filter_value_2 = {"id": 1}
        expected_filter_clause = """JSON_VALUE(langchain_vector_store_tests.
        content_metadata, :JSON_VALUE_1) = :JSON_VALUE_2"""
        mocks["_handle_field_filter"].return_value = expected_filter_clause

        filter_clause_returned = store._create_filter_clause(store, filter_value_2)
        assert filter_clause_returned == expected_filter_clause
        mocks["_handle_field_filter"].assert_called_once_with("id", 1)
        mocks["_handle_field_filter"].reset_mock()

        # filter case 3 - Filter value is not list
        filter_value_2 = {"$or": {"hi"}}

        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value_2)
        assert (
            str(context.exception)
            == """Expected a list, but got <class 'set'> for value: {'hi'}"""
        )
        mocks["_handle_field_filter"].reset_mock()

        # filter case 4 - length of fields >1 and have operator, not fields
        filter_value_4 = {"$eq": {}, "$gte": 1}
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value_4)
        assert (
            str(context.exception)
            == """Invalid filter condition. Expected a field but got: $eq"""
        )
        mocks["_handle_field_filter"].reset_mock()

        # filter case 5 - length of fields > 1 and have all fields, we AND it together

        filter_value_5 = {
            "id": {"$eq": [1, 5, 2, 9]},
            "location": {"$eq": ["pond", "market"]},
        }
        expected_filter_clause = (
            "JSON_VALUE(langchain_vector_store_tests.content_metadata, :JSON_VALUE_1)"
            " = :JSON_VALUE_2 AND "
            "JSON_VALUE(langchain_vector_store_tests.content_metadata,"
            " :JSON_VALUE_3) = :JSON_VALUE_4"
        )
        mock_sqlalchemy_and.return_value = expected_filter_clause
        store._create_filter_clause(store, filter_value_5)
        assert mocks["_handle_field_filter"].call_count == 2
        mocks["_handle_field_filter"].reset_mock()

        # filter case 6 - empty dictionary
        filter_value_6: Dict = {}
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._create_filter_clause(store, filter_value_6)
        assert str(context.exception) == """Got an empty dictionary for filters."""

        # filter case 7 - filter is None
        filter_value_7 = None
        filter_clause_returned = store._create_filter_clause(store, filter_value_7)
        assert filter_clause_returned is None


def test_handle_field_filter() -> None:
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._handle_field_filter",
        wraps=SQLServer_VectorStore._handle_field_filter,
    ), patch("sqlalchemy.and_", wraps=MagicMock) as mock_sqlalchemy_and, patch.object(
        sqlalchemy, "or_", wraps=MagicMock
    ), patch(
        "sqlalchemy.sql.operators.ne", wraps=MagicMock
    ) as mock_sqlalchemy_ne, patch(
        "sqlalchemy.sql.operators.lt", wraps=MagicMock
    ) as mock_sqlalchemy_lt, patch(
        "sqlalchemy.sql.operators.ge", wraps=MagicMock
    ) as mock_sqlalchemy_gte, patch(
        "sqlalchemy.sql.operators.le",
        wraps=MagicMock,
    ) as mock_sqlalchemy_lte:
        # Test case 1: field startWith $
        field = "$AND"
        value = 1
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._handle_field_filter(store, field, value)
        assert (
            str(context.exception)
            == "Invalid filter condition. Expected a field but got an operator: $AND"
        )

        # Test case 2: field is not valid identifier
        field = "/?"
        value = 1
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._handle_field_filter(store, field, value)
        assert (
            str(context.exception)
            == f"Invalid field name: {field}. Expected a valid identifier."
        )

        # Test case 3: more than 1 filter for value
        field = "id"
        value_1 = {"id": "3", "name": "john"}
        expected_message = (
            "Invalid filter condition."
            " Expected a value which is a dictionary with a single key "
            "that corresponds to an operator but got a dictionary with 2 keys."
            " The first few keys are: "
            "['id', 'name']"
        )
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._handle_field_filter(store, field, value_1)

        assert str(context.exception) == expected_message

        # Test case 4: field is not valid identifier
        field = "id"
        value_2 = {"$neee": 1}
        with unittest.TestCase().assertRaises(ValueError) as context:
            store._handle_field_filter(store, field, value_2)
        assert str(context.exception).startswith("Invalid operator: $neee.")

        # Test case 5: SPECIAL CASED OPERATORS
        field = "id"
        value_3 = {"$ne": 1}
        expected_response = """JSON_VALUE(langchain_vector_store_tests.content_metadata,
          :JSON_VALUE_1) != :JSON_VALUE_2"""
        mock_sqlalchemy_ne.return_value = expected_response
        handle_field_filter_response = store._handle_field_filter(store, field, value_3)
        assert handle_field_filter_response == expected_response

        # Test case 6: NUMERIC OPERATORS
        field = "id"
        value_4 = {"$lt": 1}
        expected_response = """JSON_VALUE(langchain_vector_store_tests.content_metadata,
          :JSON_VALUE_1) < :JSON_VALUE_2"""
        mock_sqlalchemy_lt.return_value = expected_response
        handle_field_filter_response = store._handle_field_filter(store, field, value_4)
        assert handle_field_filter_response == expected_response

        # Test case 7: BETWEEN OPERATOR
        field = "id"
        value_5 = {"$between": (1, 2)}
        expected_response = """CAST(JSON_VALUE(
        langchain_vector_store_tests.content_metadata, :JSON_VALUE_1) AS
         NUMERIC(10, 2)) >= :param_1 AND 
         CAST(JSON_VALUE(langchain_vector_store_tests.content_metadata,
          :JSON_VALUE_1) AS NUMERIC(10, 2)) <= :param_2"""
        mock_sqlalchemy_lte.return_value = (
            "CAST(JSON_VALUE(langchain_vector_store_tests.content_metadata,"
            " :JSON_VALUE_1) AS NUMERIC(10, 2)) <= :param_2"
        )
        mock_sqlalchemy_gte.return_value = (
            "CAST(JSON_VALUE(langchain_vector_store_tests.content_metadata,"
            " :JSON_VALUE_1) AS NUMERIC(10, 2)) >= :param_1 "
        )
        mock_sqlalchemy_and.return_value = expected_response
        handle_field_filter_response = store._handle_field_filter(store, field, value_5)
        assert handle_field_filter_response == expected_response
        mock_sqlalchemy_and.assert_called_once_with(
            mock_sqlalchemy_gte.return_value, mock_sqlalchemy_lte.return_value
        )

        # Test case 8: SPECIAL CASED OPERATOR unsupported
        field = "id"
        value_6: Dict = {"$in": [[], []]}
        with unittest.TestCase().assertRaises(NotImplementedError) as context_n:
            store._handle_field_filter(store, field, value_6)
        assert (
            str(context_n.exception) == "Unsupported type: <class 'list'> for value: []"
        )

        # Test case 9: SPECIAL CASED OPERATOR IN
        field = "id"
        value_7 = {"$in": ["adam", "bob"]}
        expected_response = (
            "JSON_VALUE(:JSON_VALUE_1, :JSON_VALUE_2) IN (__[POSTCOMPILE_JSON_VALUE_3])"
        )
        handle_field_filter_response = store._handle_field_filter(store, field, value_7)
        assert str(handle_field_filter_response) == expected_response

        # Test case 10: SPECIAL CASED OPERATOR LIKE
        field = "id"
        value_8 = {"$like": ["adam", "bob"]}
        expected_response = (
            "JSON_VALUE(:JSON_VALUE_1, :JSON_VALUE_2) LIKE :JSON_VALUE_3"
        )
        handle_field_filter_response = store._handle_field_filter(store, field, value_8)
        assert str(handle_field_filter_response) == expected_response


def test_docs_from_result() -> None:
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._docs_from_result",
        wraps=SQLServer_VectorStore._docs_from_result,
    ):
        result = [
            (
                Document(
                    page_content="id 3",
                    metadata={
                        "name": "jane",
                        "date": "2021-01-01",
                        "count": 3,
                        "is_active": True,
                        "tags": ["b", "d"],
                        "location": [3.0, 4.0],
                        "id": 3,
                        "height": 2.4,
                        "happiness": None,
                    },
                ),
                0.982679262929245,
            ),
            (
                Document(
                    page_content="id 1",
                    metadata={
                        "name": "adam",
                        "date": "2021-01-01",
                        "count": 1,
                        "is_active": True,
                        "tags": ["a", "b"],
                        "location": [1.0, 2.0],
                        "id": 1,
                        "height": 10.0,
                        "happiness": 0.9,
                        "sadness": 0.1,
                    },
                ),
                1.0078365850902349,
            ),
        ]
        expected_documents = [
            Document(
                page_content="id 3",
                metadata={
                    "name": "jane",
                    "date": "2021-01-01",
                    "count": 3,
                    "is_active": True,
                    "tags": ["b", "d"],
                    "location": [3.0, 4.0],
                    "id": 3,
                    "height": 2.4,
                    "happiness": None,
                },
            ),
            Document(
                page_content="id 1",
                metadata={
                    "name": "adam",
                    "date": "2021-01-01",
                    "count": 1,
                    "is_active": True,
                    "tags": ["a", "b"],
                    "location": [1.0, 2.0],
                    "id": 1,
                    "height": 10.0,
                    "happiness": 0.9,
                    "sadness": 0.1,
                },
            ),
        ]
        documents_returned = store._docs_from_result(store, result)

        assert documents_returned == expected_documents


def test_docs_and_scores_from_result() -> None:
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._docs_and_scores_from_result",
        wraps=SQLServer_VectorStore._docs_and_scores_from_result,
    ):
        result = [
            SimpleNamespace(
                EmbeddingStore=SimpleNamespace(
                    content="hi", content_metadata={"key": "value"}
                ),
                distance=1,
            )
        ]

        expected_documents = [
            (Document(page_content="hi", metadata={"key": "value"}), 1)
        ]
        resulted_docs_and_score = store._docs_and_scores_from_result(store, result)

        assert resulted_docs_and_score == expected_documents


def test_delete() -> None:
    store, mocks = generalized_mock_factory()

    with patch(
        "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.delete",
        wraps=SQLServer_VectorStore.delete,
    ), patch.object(store, "_delete_texts_by_ids", wraps=mocks["_delete_texts_by_ids"]):
        ids: Optional[list[int]] = None
        assert store.delete(store, ids) is False

        ids = []
        assert store.delete(store, ids) is False

        ids = [1, 2, 3]
        mocks["_delete_texts_by_ids"].return_value = 0
        assert store.delete(store, ids) is False

        ids = [1, 2, 3]
        mocks["_delete_texts_by_ids"].return_value = 1
        assert store.delete(store, ids) is True
