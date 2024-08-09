from collections import defaultdict
from typing import List

from haystack import Document, component
from haystack.document_stores.types import DocumentStore


@component
class AutoMergingRetriever:
    """
    A retriever which returns parent documents of the matched leaf nodes documents, based on a threshold setting.

    The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
    are indexed in a document store. During retrieval, if the number of matched leaf documents below the same parent is
    higher than a defined threshold, the retriever will return the parent document instead of the individual leaf
    documents.

    The rational is, given that a paragraph is split into multiple chunks represented as leaf documents, and if for
    a given query, multiple chunks are matched, the whole paragraph might be more informative than the individual
    chunks alone.
    """

    def __init__(self, document_store: DocumentStore, threshold: float = 0.9):
        """
        Initialize the AutoMergingRetriever.

        Groups the matched leaf documents by their parent documents and returns the parent documents if the number of
        matched leaf documents below the same parent is higher than the defined threshold. Otherwise, returns the
        matched leaf documents.

        :param document_store: DocumentStore from which to retrieve the parent documents
        :param threshold: Threshold to decide whether to return the parent document instead of the individual leaf
                          documents
        """

        if threshold > 1 or threshold < 0:
            raise ValueError("The threshold parameter must be between 0 and 1.")

        self.document_store = document_store
        self.threshold = threshold

    @component.output_types(documents=List[Document])
    def run(self, matched_leaf_documents: List[Document]):
        """
        Run the AutoMergingRetriever.

        :param matched_leaf_documents: List of leaf documents that were matched by a retriever
        """

        docs_to_return = []

        # group the matched leaf documents by their parent documents
        parent_documents = defaultdict(int)
        for doc in matched_leaf_documents:
            parent_documents[doc.parent_id] = parent_documents.get(doc.parent_id, 0) + 1

        # find total number of children for each parent document
        for doc in parent_documents.keys():
            parent_doc = self.document_store.filter_documents({"field": "id", "operator": "==", "value": doc.parent_id})
            parent_children_count = parent_doc[0].children_count

            # return either the parent document or the matched leaf documents based on the threshold value
            if parent_children_count / len(matched_leaf_documents) >= self.threshold:
                docs_to_return.append(parent_doc[0])
            else:
                docs_to_return.append([doc for doc in matched_leaf_documents if doc.parent_id == doc])
