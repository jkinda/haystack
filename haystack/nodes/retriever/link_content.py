import logging
from datetime import datetime
from http import HTTPStatus
from typing import Optional, Dict, List, Union, Callable, Any, Tuple
from urllib.parse import urlparse

import requests
from boilerpy3 import extractors
from requests import Response
from requests.exceptions import InvalidURL

from haystack import __version__
from haystack.nodes import PreProcessor, BaseComponent
from haystack.schema import Document, MultiLabel

logger = logging.getLogger(__name__)


def html_content_handler(response: Response, raise_on_failure: bool = False) -> Optional[str]:
    """
    Extracts content from the response text using the boilerpy3 extractor.
    """
    extractor = extractors.ArticleExtractor(raise_on_failure=raise_on_failure)
    content = ""
    try:
        content = extractor.get_content(response.text)
    except Exception as e:
        if raise_on_failure:
            raise e
        logger.debug("Couldn't extract content from %s", response.url)
    return content


def pdf_content_handler(response: Response, raise_on_failure: bool = False) -> Optional[str]:
    # TODO: implement this
    return None


class LinkContentRetriever(BaseComponent):
    """
    LinkContentRetriever fetches content from a URL and converts it into a list of Document objects.

    LinkContentRetriever supports the following content types:
    - HTML

    """

    outgoing_edges = 1

    REQUEST_HEADERS = {
        "accept": "*/*",
        "User-Agent": f"haystack/LinkContentRetriever/{__version__}",
        "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
        "referer": "https://www.google.com/",
    }

    def __init__(self, processor: Optional[PreProcessor] = None, raise_on_failure: Optional[bool] = False):
        """

        Creates a LinkContentRetriever instance.
        :param processor: PreProcessor to apply to the extracted text
        :param raise_on_failure: A boolean indicating whether to raise an exception when a failure occurs
                         during content extraction. If False, the error is simply logged and the program continues.
                         Defaults to False.
        """
        super().__init__()
        self.processor = processor
        self.raise_on_failure = raise_on_failure
        self.handlers: Dict[str, Callable] = {"html": html_content_handler, "pdf": pdf_content_handler}

    def fetch(self, url: str, timeout: Optional[int] = 3, doc_kwargs: Optional[dict] = None) -> List[Document]:
        """
        Fetches content from a URL and converts it into a list of Document objects.
        :param url: URL to fetch content from.
        :param timeout: Timeout in seconds for the request.
        :param doc_kwargs: Optional kwargs to pass to the Document constructor.
        :return: List of Document objects.
        """
        if not url or not self._is_valid_url(url):
            raise InvalidURL("Invalid or missing URL: {}".format(url))

        doc_kwargs = doc_kwargs or {}
        extracted_doc = {"url": url, "meta": {"timestamp": int(datetime.utcnow().timestamp())}}
        extracted_doc.update(doc_kwargs)

        response = self._get_response(url, timeout=timeout)
        has_content = response.status_code == HTTPStatus.OK and response.text
        fetched_documents = []
        if has_content:
            handler = "html"  # will handle non-HTML content types soon, add content type resolution here
            if handler in self.handlers:
                extracted_content = self.handlers[handler](response, self.raise_on_failure)
                if extracted_content:
                    extracted_doc["content"] = extracted_content
                    logger.debug("%s handler extracted content from %s", handler, url)
                else:
                    text = extracted_doc.get("text", "")  # we might have snippet under text, use it as fallback
                    logger.debug("%s handler failed to extract content from %s", handler, url)
                    extracted_doc["content"] = text
                document = Document.from_dict(extracted_doc)

                if self.processor:
                    fetched_documents = self.processor.process(documents=[document])
                else:
                    fetched_documents = [document]
        return fetched_documents

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        """
        Fetches content from a URL specified by query parameter and converts it into a list of Document objects.

        param query: The query - a URL to fetch content from.
        param filters: Not used.
        param top_k: Return only the top_k results. If None, the top_k value passed to the constructor is used.
        param index: Not used.
        param headers: Not used.
        param scale_score: Not used.
        param document_store: Not used.

        return: List of Document objects.
        """
        if not query:
            raise ValueError("LinkContentRetriever run requires the `query` parameter")
        documents = self.fetch(url=query)
        return {"documents": documents}, "output_1"

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        """
        Takes a list of queries, where each query is expected to be a URL. For each query, the method
        fetches content from the specified URL and transforms it into a list of Document objects. The output is a list
        of these document lists, where each individual list of Document objects corresponds to the content retrieved
        from a specific query URL.

        param queries: List of queries - URLs to fetch content from.
        param filters: Not used.
        param top_k: Return only the top_k results. If None, the top_k value passed to the constructor is used.
        param index: Not used.
        param headers: Not used.
        param batch_size: Not used.
        param scale_score: Not used.
        param document_store: Not used.

        return: List of lists of Document objects.
        """
        results = []
        if isinstance(queries, str):
            queries = [queries]
        elif not isinstance(queries, list):
            raise ValueError(
                "LinkContentRetriever run_batch requires the `queries` parameter to be Union[str, List[str]]"
            )
        for query in queries:
            results.append(self.fetch(url=query))

        return {"documents": results}, "output_1"

    def _get_response(self, url: str, timeout: Optional[int]) -> requests.Response:
        """
        Fetches content from a URL. Returns a response object.
        :param url: The URL to fetch content from.
        :param timeout: The timeout in seconds.
        :return: A response object.
        """
        try:
            response = requests.get(url, headers=LinkContentRetriever.REQUEST_HEADERS, timeout=timeout)
            if response.status_code != HTTPStatus.OK:
                logger.debug("Couldn't retrieve content from %s, status code: %s", url, response.status_code)
        except requests.RequestException as e:
            if self.raise_on_failure:
                raise e

            logger.debug("Couldn't retrieve content from %s", url)
            return requests.Response()
        return response

    def _is_valid_url(self, url: str) -> bool:
        """
        Checks if a URL is valid.

        :param url: The URL to check.
        :return: True if the URL is valid, False otherwise.
        """

        result = urlparse(url)
        # schema is http or https and netloc is not empty
        return all([result.scheme in ["http", "https"], result.netloc])
