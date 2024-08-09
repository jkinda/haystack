import pytest

from haystack import Document
from haystack.components.builders import HierarchicalDocumentBuilder
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestSentenceWindowRetriever:
    def test_init_default(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        assert retriever.threshold == 0.5

    def test_init_with_parameters(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore(), threshold=0.7)
        assert retriever.threshold == 0.7

    def test_init_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            AutoMergingRetriever(InMemoryDocumentStore(), threshold=-2)

    def test_run(self):
        docs = [Document(content="The monarch of the wild blue yonder rises from the eastern side of the horizon.")]
        builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2], split_overlap=0, split_by="word")
        docs = builder.run(docs)

        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["children_ids"]:
                doc_store_parents.write_documents([doc])

        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["children_ids"]]
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)
        retriever.run(leaf_docs[3:5])
