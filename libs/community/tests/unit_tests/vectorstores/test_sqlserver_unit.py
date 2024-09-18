import os
from unittest import TestCase, mock
from unittest.mock import Mock

import pytest
from langchain_core.documents.base import Document
from sqlalchemy import UUID
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.sqlserver import (
    SQLServer_VectorStore,
)

_CONNECTION_STRING_WITH_UID_AND_PWD = str(
    os.environ.get("TEST_AZURESQLSERVER_CONNECTION_STRING_WITH_UID")
)
_CONNECTION_STRING_WITH_TRUSTED_CONNECTION = str(
    os.environ.get("TEST_AZURESQLSERVER_TRUSTED_CONNECTION")
)
_ENTRA_ID_CONNECTION_STRING_NO_PARAMS = str(
    os.environ.get("TEST_ENTRA_ID_CONNECTION_STRING_NO_PARAMS")
)
_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO = str(
    os.environ.get("TEST_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO")
)
_TABLE_NAME = "langchain_vector_store_tests"
_SCHEMA_ = "langchain_vector_store_tests"
EMBEDDING_LENGTH = 1536


@pytest.fixture
def mock_session():
    return Mock(spec=Session)


# We need to mock this so that actual connection is not attempted
# after mocking _provide_token.
@mock.patch("sqlalchemy.dialects.mssql.dialect.initialize")
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._provide_token"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._prepare_json_data_type"
)
def test_that_given_a_valid_entra_id_connection_string_entra_id_authentication_is_used(
    prep_data_type: Mock,
    provide_token: Mock,
    dialect_initialize: Mock,
) -> None:
    """Test that if a valid entra_id connection string is passed in
    to SQLServer_VectorStore object, entra id authentication is used
    and connection is successful."""

    # Connection string is of the form below.
    # "mssql+pyodbc://lc-test.database.windows.net,1433/lcvectorstore
    # ?driver=ODBC+Driver+17+for+SQL+Server"
    SQLServer_VectorStore(
        connection_string=_ENTRA_ID_CONNECTION_STRING_NO_PARAMS,
        embedding_length=EMBEDDING_LENGTH,
        # FakeEmbeddings returns embeddings of the same size as `embedding_length`.
        embedding_function=FakeEmbeddings(size=EMBEDDING_LENGTH),
        table_name=_TABLE_NAME,
    )
    # _provide_token is called only during Entra ID authentication.
    provide_token.assert_called()

    # reset the mock so that it can be reused.
    provide_token.reset_mock()

    # "mssql+pyodbc://lc-test.database.windows.net,1433/lcvectorstore
    # ?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=no"
    SQLServer_VectorStore(
        connection_string=_ENTRA_ID_CONNECTION_STRING_TRUSTED_CONNECTION_NO,
        embedding_length=EMBEDDING_LENGTH,
        # FakeEmbeddings returns embeddings of the same size as `embedding_length`.
        embedding_function=FakeEmbeddings(size=EMBEDDING_LENGTH),
        table_name=_TABLE_NAME,
    )
    provide_token.assert_called()


# We need to mock this so that actual connection is not attempted
# after mocking _provide_token.
@mock.patch("sqlalchemy.dialects.mssql.dialect.initialize")
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._provide_token"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._prepare_json_data_type"
)
def test_that_given_a_connection_string_with_uid_and_pwd_entra_id_auth_is_not_used(
    prep_data_type: Mock,
    provide_token: Mock,
    dialect_initialize: Mock,
) -> None:
    """Test that if a connection string is provided to SQLServer_VectorStore object,
    and connection string has username and password, entra id authentication is not
    used and connection is successful."""

    connect_to_vector_store(_CONNECTION_STRING_WITH_UID_AND_PWD)

    provide_token.assert_not_called()


# We need to mock this so that actual connection is not attempted
# after mocking _provide_token.
@mock.patch("sqlalchemy.dialects.mssql.dialect.initialize")
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._provide_token"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._prepare_json_data_type"
)
def test_that_connection_string_with_trusted_connection_yes_does_not_use_entra_id_auth(
    prep_data_type: Mock,
    provide_token: Mock,
    dialect_initialize: Mock,
) -> None:
    """Test that if a connection string is provided to SQLServer_VectorStore object,
    and connection string has `trusted_connection` set to `yes`, entra id
    authentication is not used and connection is successful."""

    # Connection string is of the form below.
    # mssql+pyodbc://@lc-test.database.windows.net,1433/lcvectorstore
    # ?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)

    # _provide_token is called only during Entra ID authentication.
    provide_token.assert_not_called()


# We need to mock this so that actual connection is not attempted
# creates engine and table
# after mocking _provide_token.
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._get_embedding_store"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_table_if_not_exists"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._provide_token"
)
def test_that_connection_string_with_trusted_connection_yes_creates_engine_and_table(
    provide_token: Mock,
    _create_engine: Mock,
    _create_table_if_not_exists: Mock,
    _get_embedding_store: Mock,
) -> None:
    """Test that if a connection string is provided to SQLServer_VectorStore object,
    and connection string has `trusted_connection` set to `yes`, entra id
    authentication is not used and connection is successful."""

    # Connection string is of the form below.
    # mssql+pyodbc://@lc-test.database.windows.net,1433/lcvectorstore
    # ?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    SQLServer_VectorStore(
        connection_string=_CONNECTION_STRING_WITH_TRUSTED_CONNECTION,
        embedding_length=EMBEDDING_LENGTH,
        # FakeEmbeddings returns embeddings of the same size as `embedding_length`.
        embedding_function=FakeEmbeddings(size=EMBEDDING_LENGTH),
        table_name=_TABLE_NAME,
    )

    # _provide_token is called only during Entra ID authentication.
    provide_token.assert_not_called()
    _create_engine.assert_called_once()
    _get_embedding_store.assert_called_once()
    _create_table_if_not_exists.assert_called_once()


@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search_by_vector"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._get_embedding_store"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_table_if_not_exists"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._provide_token"
)
def test_similarity_search(
    provide_token: Mock,
    _create_engine: Mock,
    _create_table_if_not_exists: Mock,
    _get_embedding_store: Mock,
    similarity_search_by_vector: Mock,
) -> None:
    """Test that similarity search uses the given input to get
    embeddings and call similarity_search_by_vector with query,
    k and embedding values"""

    query = "hi"
    mock_responses = {"hello": [0.1, 0.2, 0.3], "hi": [0.01, 0.02, 0.03]}
    store = connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)

    store.embedding_function = Mock()
    # Set the side effect to return values from mock_responses based on the input
    store.embedding_function.embed_query.side_effect = lambda query: mock_responses[
        query
    ]
    store.similarity_search(query)
    similarity_search_by_vector.assert_called_once_with(
        store.embedding_function.embed_query("hi"), 4
    )

    similarity_search_by_vector.reset_mock()

    store.similarity_search(query, 7)
    similarity_search_by_vector.assert_called_once_with(
        store.embedding_function.embed_query("hi"), 7
    )

    similarity_search_by_vector.reset_mock()


@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._docs_from_result"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search_by_vector_with_score"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._get_embedding_store"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_table_if_not_exists"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._provide_token"
)
def test_similarity_search_by_vector(
    provide_token: Mock,
    _create_engine: Mock,
    _create_table_if_not_exists: Mock,
    _get_embedding_store: Mock,
    similarity_search_by_vector_with_score: Mock,
    _docs_from_result: Mock,
) -> None:
    """Test that similarity search uses the given input to get
    embeddings and call similarity_search_by_vector with query,
    k and embedding values"""
    embeddings = ["0.01", "0.02", "0.03"]
    mock_responses = {
        tuple(["0.01", "0.02", "0.03"]): (
            Document(page_content="""Got these on sale for roughly 25 cents per cup"""),
            0.9588668232580106,
        ),
    }

    store = connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)
    store.embedding_function = Mock()
    # Set the side effect to return values from mock_responses based on the input
    store.similarity_search_by_vector_with_score.side_effect = (
        lambda embeddings, num: mock_responses.get(
            tuple(embeddings), "default_response"
        )
    )
    store.similarity_search_by_vector(embeddings)
    expected_result = (
        Document(page_content="Got these on sale for roughly 25 cents per cup"),
        0.9588668232580106,
    )
    _docs_from_result.assert_called_once_with(expected_result)

    similarity_search_by_vector_with_score.reset_mock()
    _docs_from_result.reset_mock()

    store.similarity_search_by_vector(embeddings, 7)
    similarity_search_by_vector_with_score.assert_called_once_with(embeddings, 7)
    _docs_from_result.assert_called_once_with(expected_result)


@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore.similarity_search_by_vector_with_score"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._get_embedding_store"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_table_if_not_exists"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._provide_token"
)
def test_similarity_search_with_score(
    provide_token: Mock,
    _create_engine: Mock,
    _create_table_if_not_exists: Mock,
    _get_embedding_store: Mock,
    similarity_search_by_vector_with_score: Mock,
) -> None:
    """Test that similarity search uses the given input to get
    embeddings and call similarity_search_by_vector with query,
    k and embedding values"""

    query = "hi"
    mock_responses = {"hello": [0.1, 0.2, 0.3], "hi": [0.01, 0.02, 0.03]}
    store = connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)

    store.embedding_function = Mock()
    # Set the side effect to return values from mock_responses based on the input
    store.embedding_function.embed_query.side_effect = lambda query: mock_responses[
        query
    ]
    store.similarity_search_with_score(query)
    similarity_search_by_vector_with_score.assert_called_once_with(
        store.embedding_function.embed_query("hi"), 4
    )

    similarity_search_by_vector_with_score.reset_mock()

    store.similarity_search_with_score(query, 7)
    similarity_search_by_vector_with_score.assert_called_once_with(
        store.embedding_function.embed_query("hi"), 7
    )


@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._docs_and_scores_from_result"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._search_store"
)
def test_similarity_search_by_vector_with_score(
    _search_store: Mock,
    _docs_and_scores_from_result: Mock,
) -> None:
    """Test that similarity search uses the given input to get
    embeddings and call similarity_search_by_vector with query,
    k and embedding values"""

    embeddings = [0.01, 0.02, 0.03]
    mock_responses_search = {
        tuple([0.01, 0.02, 0.03]): """(<langchain_community.vectorstores.sqlserver.
    SQLServer_VectorStore._get_embedding_store.<locals>.EmbeddingStore object
      at 0x0000025EFFF84810>,
          0.9595672912317021)"""
    }
    mock_responses = {
        """(<langchain_community.vectorstores.sqlserver.
    SQLServer_VectorStore._get_embedding_store.<locals>.EmbeddingStore object
      at 0x0000025EFFF84810>,
          0.9595672912317021)""": (
            Document(page_content="""Got these on sale for roughly 25 cents per cup"""),
            0.9588668232580106,
        ),
    }
    expected_search_result = """(<langchain_community.vectorstores.sqlserver.
    SQLServer_VectorStore._get_embedding_store.<locals>.EmbeddingStore object
      at 0x0000025EFFF84810>,
          0.9595672912317021)"""
    expected_docs = (
        Document(page_content="""Got these on sale for roughly 25 cents per cup"""),
        0.9588668232580106,
    )

    store = connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)

    store._search_store.side_effect = lambda embeddings, num: mock_responses_search.get(
        tuple(embeddings), "default_response"
    )
    store._docs_and_scores_from_result.side_effect = (
        lambda embeddings: mock_responses.get(
            expected_search_result, "default_response"
        )
    )
    docs_result = store.similarity_search_by_vector_with_score(embeddings)
    _search_store.assert_called_once_with(embeddings, 4)
    _docs_and_scores_from_result.assert_called_once_with(expected_search_result)

    assert docs_result == expected_docs


@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._insert_embeddings"
)
def test_add_texts(_insert_embeddings: Mock) -> None:
    """Test that similarity search uses the given input to get
    embeddings and call similarity_search_by_vector with query,
    k and embedding values"""

    texts = ["hi", "hello", "welcome"]
    embeddings = [0.01, 0.02, 0.03]
    metadatas = [
        {"id": 1, "summary": "Good Quality Dog Food"},
        {"id": 2, "summary": "Nasty No flavor"},
        {"id": 3, "summary": "stale product"},
    ]
    ids = [1, 2, 3]
    expected_ids = [
        UUID("f7509836-f0fc-4b21-ac0c-90ae901073fc"),
        UUID("ac7029ee-6b3a-400f-8b66-d8ea621234b7"),
        UUID("704396df-8c0a-4904-a064-71d1c0bb722f"),
    ]

    mock_embeddings_responses = {tuple(["hi", "hello", "welcome"]): embeddings}
    mock_responses = {
        tuple(embeddings): [
            UUID("f7509836-f0fc-4b21-ac0c-90ae901073fc"),
            UUID("ac7029ee-6b3a-400f-8b66-d8ea621234b7"),
            UUID("704396df-8c0a-4904-a064-71d1c0bb722f"),
        ]
    }
    store = connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)

    store.embedding_function = Mock()
    # Set the side effect to return values from mock_responses based on the input
    store.embedding_function.embed_documents.side_effect = (
        lambda texts: mock_embeddings_responses.get(tuple(texts), "default_response")
    )

    store._insert_embeddings.side_effect = (
        lambda texts, embeddings, metadatas, ids: mock_responses.get(
            tuple(embeddings), "default_response"
        )
    )

    returned_ids = store.add_texts(texts, metadatas, ids)
    store.embedding_function.embed_documents.assert_called_once_with(texts)

    _insert_embeddings.assert_called_once_with(
        texts, store.embedding_function.embed_documents(texts), metadatas, ids
    )
    assert str(expected_ids) == str(returned_ids)


def test_drop_failure(mock_session):
    # Create an instance of the mock session
    mock_session = mock_session.return_value
    # Mock the methods used in the drop function
    mock_bind = mock.Mock()
    mock_session.get_bind.return_value = mock_bind
    mock_session.commit.side_effect = ProgrammingError("mock error", None, None)

    # Create an instance of SQLServer_VectorStore
    store = connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)
    store._bind = mock_bind
    store._embedding_store = mock.Mock()
    store._embedding_store.__table__ = mock.Mock()
    store._embedding_store.__table__.drop.side_effect = ProgrammingError(
        "mock error", None, None
    )

    # Call the drop method and assert it handles the exception
    with TestCase().assertLogs(level="ERROR") as log:
        store.drop()
        assert "Unable to drop vector store." in log.output[0]

    # Assert the expected behavior
    store._embedding_store.__table__.drop.assert_called_once_with(mock_bind)
    mock_session.commit.assert_not_called()


@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._get_embedding_store",
    autospec=True,
)
@mock.patch(
    "sqlalchemy.orm.session.Session.query",
)
@mock.patch(
    "sqlalchemy.orm.query.Query.filter",
)
@mock.patch(
    "sqlalchemy.orm.query.Query.order_by",
)
@mock.patch(
    "sqlalchemy.orm.query.Query.limit",
)
@mock.patch(
    "sqlalchemy.orm.query.Query.all",
)
def test_search_store_success(
    query_all, query_limit, query_order_by, query_filter, mock_query, mock_embedding
):
    mock_query.return_value.filter.return_value.order_by.return_value.all.return_value = [
        "query_with_all"
    ]
    store = connect_to_vector_store(_CONNECTION_STRING_WITH_TRUSTED_CONNECTION)

    store._embedding_store.return_value = mock_embedding
    store.distance_strategy = "cosine"

    # Arrange
    embedding = [0.1, 0.2, 0.3]
    k = 4

    # Act
    results = store._search_store(embedding, k)

    mock_query.assert_called_once()
    # Assert
    assert results == mock_query.return_value.filter().order_by().limit().all()


@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._get_embedding_store"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_table_if_not_exists"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._create_engine"
)
@mock.patch(
    "langchain_community.vectorstores.sqlserver.SQLServer_VectorStore._provide_token"
)
def connect_to_vector_store(  # provide_token: Mock,
    provide_token: Mock,
    _create_engine: Mock,
    _create_table_if_not_exists: Mock,
    _get_embedding_store: Mock,
    conn_string: str,
) -> SQLServer_VectorStore:
    store = SQLServer_VectorStore(
        connection_string=conn_string,
        embedding_length=EMBEDDING_LENGTH,
        # FakeEmbeddings returns embeddings of the same size as `embedding_length`.
        embedding_function=FakeEmbeddings(size=EMBEDDING_LENGTH),
        table_name=_TABLE_NAME,
    )

    return store


def setup():
    setup_vector_store()


def setup_vector_store():

    
    SQLServer_VectorStore = mock.Mock()
