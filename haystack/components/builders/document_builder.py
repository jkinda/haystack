from typing import Any, Dict, List

from haystack import Document, component


@component
class DocumentsBuilder:
    """
    A component that allows updating a Documents metadata.
    """

    @component.output_types(documents=List[Document])
    def run(self, metadata: List[List[Dict[str, Any]]], documents: List[Document]):
        """
        Update the metadata of the documents.

        :param metadata:
        :type metadata:
        :param documents:
        :type documents:
        :return:
        :rtype:
        """
        all_documents = []
        for meta, document in zip(metadata, documents):
            documents = [
                Document(content=document.content, metadata={**document.meta, **meta}) for document in documents
            ]
            all_documents.append(documents)
        return {"documents": all_documents}
