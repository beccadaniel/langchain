from typing import Any, Iterable, List, Optional, Type, Union

from azure.identity import DefaultAzureCredential
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
from sqlalchemy import Column, Uuid, bindparam, create_engine, event, text
from sqlalchemy.dialects.mssql import JSON, NVARCHAR, VARBINARY, VARCHAR
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

import json
import logging
import struct
import uuid

Base = declarative_base()  # type: Any

_embedding_store: Any = None

AZURE_TOKEN_URL = "https://database.windows.net/.default" # Token URL for Azure DBs.
EXTRA_PARAMS = ";Trusted_Connection=Yes"
SQL_COPT_SS_ACCESS_TOKEN = 1256 # Connection option defined by microsoft in msodbcsql.h

class SQLServer_VectorStore(VectorStore):
    """SQL Server Vector Store.

    This class provides a vector store interface for adding texts and performing
        similarity searches on the texts in SQL Server.

    """

    def __init__(
        self,
        *,
        connection: Union[Engine, str],
        embedding_function: Embeddings,
        table_name: str,
    ) -> None:
        """Initialize the SQL Server vector store.

        Args:
            connection: SQLServer connection string or an `Engine` object.
            embedding_function: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            table_name: The name of the table to use for storing embeddings.

        """

        self.embedding_function = embedding_function
        self.table_name = table_name
        self._bind: Engine = (
            connection if isinstance(connection, Engine)
            else self._create_engine(connection)
        )
        self.EmbeddingStore = self._get_embedding_store(table_name)
        self._create_table_if_not_exists()

    def _create_engine(self, connection_url: str) -> Engine:
        return create_engine(url=connection_url, echo=True)

    def _create_table_if_not_exists(self) -> None:
        logging.info("Creating table %s", self.table_name)
        with Session(self._bind) as session:
            Base.metadata.create_all(session.get_bind())

    def _get_embedding_store(self, name: str) -> Any:
        global _embedding_store
        if _embedding_store is not None:
            return _embedding_store

        class EmbeddingStore(Base):
            """This is the base model for SQL vector store."""

            __tablename__ = name
            id = Column(Uuid, primary_key=True, default=uuid.uuid4)
            custom_id = Column(VARCHAR, nullable=True)  # column for user defined ids.
            query_metadata = Column(JSON, nullable=True)
            query = Column(NVARCHAR, nullable=False)  # defaults to NVARCHAR(MAX)
            embeddings = Column(VARBINARY, nullable=False)

        _embedding_store = EmbeddingStore
        return _embedding_store

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from texts and embeddings."""
        return super().from_texts(texts, embedding, metadatas, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        # placeholder
        return []

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Compute the embeddings for the input texts and store embeddings
            in the vectorstore.

        Args:
            texts: Iterable of strings to add into the vectorstore.
            metadatas: List of metadatas (python dicts) associated with the input texts.
            ids: List of IDs for the input texts.
            kwargs: vectorstore specific parameters.

        Returns:
            List of IDs generated from adding the texts into the vectorstore.
        """

        # Embed the texts passed in.
        embedded_texts = self.embedding_function.embed_documents(list(texts))

        # Insert the embedded texts in the vector store table.
        return self._insert_embeddings(texts, embedded_texts, metadatas, ids)

    def _insert_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert the embeddings and the texts in the vectorstore.

        Args:
            texts: Iterable of strings to add into the vectorstore.
            embeddings: List of list of embeddings.
            metadatas: List of metadatas (python dicts) associated with the input texts.
            ids: List of IDs for the input texts.
            kwargs: vectorstore specific parameters.

        Returns:
            List of IDs generated from adding the texts into the vectorstore.
        """

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if ids is None:
            ids = [metadata.pop("id", uuid.uuid4()) for metadata in metadatas]

        try:
            with Session(self._bind) as session:
                documents = []
                for idx, query in enumerate(texts):
                    # For a query, if there is no corresponding ID,
                    # we generate a uuid and add it to the list of IDs to be returned.
                    if idx < len(ids):
                        id = ids[idx]
                    else:
                        ids.append(str(uuid.uuid4()))
                        id = ids[-1]
                    embedding = embeddings[idx]
                    metadata = metadatas[idx] if idx < len(metadatas) else None

                    # Construct text, embedding, metadata as EmbeddingStore model
                    # to be inserted into the table.
                    sqlquery = text(
                        "select JSON_ARRAY_TO_VECTOR (:embeddingvalues)"
                    ).bindparams(
                        bindparam(
                            "embeddingvalues",
                            json.dumps(embedding),
                            # render the value of the parameter into SQL statement
                            # at statement execution time
                            literal_execute=True,
                        )
                    )
                    result = session.scalar(sqlquery)
                    embedding_store = self.EmbeddingStore(
                        custom_id=id,
                        query_metadata=metadata,
                        query=query,
                        embeddings=result,
                    )
                    documents.append(embedding_store)
                session.bulk_save_objects(documents)
                session.commit()
        except DBAPIError as e:
            logging.error(e.__cause__)
        return ids

    @event.listens_for(Engine, "do_connect")
    def _provide_token(dialect, conn_rec, cargs, cparams) -> None:
        """Gets token for SQLServer connection from token URL,
            and use the token to connect to the database."""
        credential = DefaultAzureCredential()

        # Remove Trusted_Connection param that SQLAlchemy adds to 
        # the connection string by default.
        cargs[0] = cargs[0].replace(EXTRA_PARAMS, "")
 
        # Create credential token
        token_bytes = credential.get_token(AZURE_TOKEN_URL).token.encode("utf-16-le")
        token_struct = struct.pack(
            f'<I{len(token_bytes)}s',
            len(token_bytes),
            token_bytes
        )      

        # Apply credential token to keyword argument
        cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: token_struct}
