from typing import Dict, List, Any, Optional

from haystack.preview import component, Document
from haystack.preview.document_stores import MemoryDocumentStore, StoreAwareMixin


@component
class MemoryRetriever(StoreAwareMixin):
    """
    A component for retrieving documents from a MemoryDocumentStore using the BM25 algorithm.

    Needs to be connected to a MemoryDocumentStore to run.
    """

    supported_stores = [MemoryDocumentStore]

    def __init__(self, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, scale_score: bool = True):
        """
        Create a MemoryRetriever component.

        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        :param scale_score: Whether to scale the BM25 score or not (default is True).

        :raises ValueError: If the specified top_k is not > 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score

    @component.output_types(documents=List[List[Document]])
    def run(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
    ):
        """
        Run the MemoryRetriever on the given input data.

        :param query: The query string for the retriever.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the BM25 scores or not.
        :param stores: A dictionary mapping document store names to instances.
        :return: The retrieved documents.

        :raises ValueError: If the specified document store is not found or is not a MemoryDocumentStore instance.
        """
        self.store: MemoryDocumentStore
        if not self.store:
            raise ValueError("MemoryRetriever needs a store to run: set the store instance to the self.store attribute")

        filters = filters if filters is not None else self.filters
        top_k = top_k if top_k is None else self.top_k
        scale_score = scale_score if scale_score is not None else self.scale_score

        docs = []
        for query in queries:
            docs.append(self.store.bm25_retrieval(query=query, filters=filters, top_k=top_k, scale_score=scale_score))
        return {"documents": docs}
