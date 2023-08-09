import logging
from pathlib import Path
from typing import Optional, List, Union, Dict

from canals.errors import PipelineRuntimeError
from tqdm import tqdm

from haystack import Document
from haystack.lazy_imports import LazyImport
from haystack.preview import component

with LazyImport("Run 'pip install farm-haystack[preprocessing]'") as langdetect_import:
    import langdetect


logger = logging.getLogger(__name__)


@component
class TextFileToDocument:
    """
    A component for converting a text file to a Document.
    """

    # @component.input
    def input(self):
        class Input:
            """
            Input data for the TextFileToDocument component.

            :param paths: A list of paths to text files.
            :param meta: Optional metadata to attach to the Documents. If a list is provided, the length of the list
                         must match the number of paths.
                         Default: `None`
            :param encoding: The encoding of the text files. Default: `"utf-8"`
            :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                          The tabular structures in documents might be noise for reader models if they
                                          don't have table parsing capability for finding answers. However, tables
                                          may also have long strings that could be possible candidates for answers.
                                          The rows containing strings are thus retained in this option.
                                          Default: `False`
            :param valid_languages: Validate languages from a list of languages specified in the [ISO 639-1 format]((https://en.wikipedia.org/wiki/ISO_639-1)).
                                    This option can be used to add a test for encoding errors. If the extracted text is
                                    not one of the valid languages, then there might be an encoding error resulting
                                    in garbled text.
                                    Default: `None`
            :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
                                 attributes. If you want to ensure you don't have duplicate Documents in your
                                 DocumentStore but texts are not unique, you can modify the metadata and pass e.g.
                                 `"meta"` to this field (for example `["content", "meta"]`).
                                 In this case the ID will be generated by using the content and the defined metadata.
                                 Default: `None`
            :param progress_bar: Whether to show a progress bar for the conversion process.
                                 Default: `True`
            """

            paths: List[Union[str, Path]]
            meta: Optional[Union[Dict, List[Dict]]]
            encoding: Optional[str]
            remove_numeric_tables: Optional[bool]
            valid_languages: Optional[List[str]]
            id_hash_keys: Optional[List[str]]
            progress_bar: Optional[bool]

        return Input

    # @component.output
    def output(self):
        class Output:
            """
            Output data from the TextFileToDocument component.

            :param documents: The converted documents.
            """

            documents: List[Document]

        return Output

    def __init__(
        self,
        encoding: str = "utf-8",
        remove_numeric_tables: bool = False,
        numeric_row_threshold: float = 0.4,
        valid_languages: Optional[List[str]] = None,
        id_hash_keys: Optional[List[str]] = None,
        progress_bar: bool = True,
    ):
        """
        Create a TextFileToDocument component.

        :param encoding: The encoding of the text files. Default: `"utf-8"`
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for reader models if they
                                      don't have table parsing capability for finding answers. However, tables
                                      may also have long strings that could be possible candidates for answers.
                                      The rows containing strings are thus retained in this option.
                                      Default: `False`
        :param numeric_row_threshold: Applicable if `remove_numeric_tables` is set to `True`. This is the threshold to
                                      determine if a line in the provided text file is a numeric table row or not.
                                      The value is the ratio of numeric words to the total number of words in a line.
        :param valid_languages: Validate languages from a list of languages specified in the [ISO 639-1 format]((https://en.wikipedia.org/wiki/ISO_639-1)).
                                This option can be used to add a test for encoding errors. If the extracted text is
                                not one of the valid languages, then there might be an encoding error resulting
                                in garbled text.
                                Default: `None`
        :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
                             attributes. If you want to ensure you don't have duplicate Documents in your DocumentStore
                             but texts are not unique, you can modify the metadata and pass e.g. `"meta"` to this field
                             (for example `["content", "meta"]`). In this case the ID will be generated by using the
                             content and the defined metadata. Default: `None`
        :param progress_bar: Whether to show a progress bar for the conversion process.
                             Default: `True`
        """
        langdetect_import.check()

        self.defaults = {
            "encoding": encoding,
            "remove_numeric_tables": remove_numeric_tables,
            "valid_languages": valid_languages,
            "id_hash_keys": id_hash_keys,
            "progress_bar": progress_bar,
        }
        self.numeric_row_threshold = numeric_row_threshold

    def run(self, data):
        """
        Convert text files to Documents.

        :param data: Input data for the TextFileToDocument component.
        """
        file_paths = data.paths
        metas = TextFileToDocument._prepare_metadata(data.meta, file_paths)

        documents = []
        for path, meta in tqdm(
            zip(file_paths, metas), total=len(file_paths), desc="Converting text files", disable=not data.progress_bar
        ):
            try:
                text = self._read_and_clean_file(
                    path=path, encoding=data.encoding, remove_numeric_tables=data.remove_numeric_tables
                )
            except Exception as e:
                logger.warning("Could not read file %s. Skipping it. Error message: %s", path, e)
                continue

            if data.valid_languages is not None and not TextFileToDocument._validate_language(
                text, data.valid_languages
            ):
                logger.warning(
                    "Text from file %s is not in one of the valid languages: %s. "
                    "The file may have been decoded incorrectly.",
                    path,
                    data.valid_languages,
                )

            document = Document(content=text, meta=meta, id_hash_keys=data.id_hash_keys)
            documents.append(document)

        return self.output(documents=documents)

    @staticmethod
    def _prepare_metadata(meta: Optional[Union[Dict, List[Dict]]], file_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Prepare the metadata for the Documents.

        :param meta: The metadata for the Documents.
        :param file_paths: The paths to the text files.
        """
        if meta is None:
            return [{"file_path": str(path)} for path in file_paths]

        if isinstance(meta, dict):
            meta = [meta] * len(file_paths)

        if len(meta) != len(file_paths):
            raise PipelineRuntimeError(
                f"The number of meta entries must match the number of paths if meta is a list. "
                f"Number of paths: {len(file_paths)}, number of meta entries: {len(meta)}."
            )

        return [{**m, "file_path": m.get("file_path", str(path))} for m, path in zip(meta, file_paths)]

    def _read_and_clean_file(self, path: Union[str, Path], encoding: str, remove_numeric_tables: bool) -> str:
        """
        Read and clean the text file.

        :param path: The path to the text file.
        :param encoding: The encoding of the text file.
        :param remove_numeric_tables: Whether to remove numeric tables.

        :return: The text of the file cleaned from numeric tables if `remove_numeric_tables` is `True`.
        """
        if not Path(path).exists():
            raise PipelineRuntimeError(f"File at path {path} does not exist.")

        with open(path, encoding=encoding) as file:
            text = file.read()
            pages = text.split("\f")
            cleaned_pages = [self._clean_page(page, remove_numeric_tables) for page in pages]
            return "\f".join(cleaned_pages)

    def _clean_page(self, page: str, remove_numeric_tables: bool) -> str:
        """
        Clean a page of text from numeric tables if `remove_numeric_tables` is `True`.

        :param page: The content of a page of a text file.
        :param remove_numeric_tables: Whether to remove numeric tables.

        :return: The text from the page cleaned from numeric tables if `remove_numeric_tables` is `True`.
        """
        cleaned_lines = page.splitlines()
        if remove_numeric_tables:
            cleaned_lines = [line for line in cleaned_lines if not self._is_numeric_row(line)]

        return "\n".join(cleaned_lines)

    def _is_numeric_row(self, line: str) -> bool:
        """
        Check if a line of a text file is a numeric row. A line is considered a numeric row if it contains more
        than 40% digits and does not end with a period.

        :param line: The content of a line of a text file.
        """
        words = line.split()
        digits = [word for word in words if any(char.isdigit() for char in word)]
        return len(digits) / len(words) > self.numeric_row_threshold and not line.strip().endswith(".")

    @staticmethod
    def _validate_language(text: str, valid_languages: List[str]) -> bool:
        """
        Validate if the detected language of the text is one of the valid languages.

        :param text: The text to validate.
        :param valid_languages: A list of valid languages.
        """
        if not valid_languages:
            return True

        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        return lang in valid_languages
