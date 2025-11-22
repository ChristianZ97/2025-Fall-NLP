from __future__ import annotations
# Enables postponed evaluation of type annotations.
# This allows the use of types that are defined later in the file
# or forward references without using string literals for type hints.

from dataclasses import dataclass, asdict
# dataclass: Decorator to automatically generate __init__, __repr__, etc., for classes.
# asdict: Utility to convert a dataclass instance (recursively) into a dictionary.

from typing import Any, Dict, List, Optional
# Any: Type that can hold any Python object.
# Dict: Dictionary type with specified key/value types.
# List: List type with specified element type.
# Optional: Indicates that a value can be either the given type or None.


@dataclass
class SentencePayload:
    """
    Represents a single sentence segment in a document.

    Attributes
    ----------
    text : str
        The raw text content of the sentence.
    """
    text: str


@dataclass
class ParagraphPayload:
    """
    Represents a paragraph, which may be further broken down into sentences
    and may contain arbitrary metadata.

    Attributes
    ----------
    text : str
        The full text content of the paragraph.
    sentences : Optional[List[SentencePayload]]
        An optional list of pre-segmented sentences that belong to this paragraph.
        If None, the paragraph is treated as a single unsegmented text block.
    metadata : Dict[str, Any]
        A dictionary for arbitrary key-value metadata associated with this paragraph.
        This can be used to store page numbers, styling info, source references, etc.
    """
    text: str
    sentences: Optional[List[SentencePayload]] = None
    # Default is None, meaning that sentence-level segmentation might not be provided.

    metadata: Dict[str, Any] = None
    # Default is None, but will be converted to an empty dict in __post_init__.

    def __post_init__(self):
        """
        Post-initialization hook automatically called after the dataclass __init__.

        Ensures that 'metadata' is always a dictionary, never None, so that
        callers do not need to check for None before using it.
        """
        if self.metadata is None:
            # Initialize metadata to an empty dictionary if not provided.
            self.metadata = {}


@dataclass
class SectionPayload:
    """
    Represents a higher-level section within a document, such as a chapter,
    heading section, or logical content block.

    Attributes
    ----------
    title : str
        The title or heading of the section.
    paragraphs : List[ParagraphPayload]
        A list of paragraphs that belong to this section.
    metadata : Dict[str, Any]
        Arbitrary metadata for the section (e.g., section number, tags, etc.).
    """
    title: str
    paragraphs: List[ParagraphPayload]
    # A section must contain one or more paragraphs.

    metadata: Dict[str, Any] = None
    # Default is None, but will be normalized to an empty dict in __post_init__.

    def __post_init__(self):
        """
        Ensures that 'metadata' is always a dictionary, not None,
        making downstream code simpler and more robust.
        """
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentPayload:
    """
    The top-level representation of a document, including its global properties
    and nested structure (sections and paragraphs).

    Attributes
    ----------
    document_id : str
        A unique identifier for the document (e.g., UUID, filename, database ID).
    title : str
        The human-readable title of the document.
    text : str
        The full raw text of the document. This can be the concatenation of
        all sections and paragraphs or a separate representation.
    metadata : Dict[str, Any]
        Arbitrary global metadata about the document (e.g., author, language,
        creation time, source system, tags, etc.).
    sections : Optional[List[SectionPayload]]
        An optional list of sections, each containing paragraphs and possibly
        further structure. If None, the document may be represented only by
        the 'text' field without explicit structural segmentation.
    """
    document_id: str
    title: str
    text: str
    metadata: Dict[str, Any]
    sections: Optional[List[SectionPayload]] = None
    # 'sections' is optional to allow simple documents with only plain text.


def payload_to_dict(payload: DocumentPayload) -> Dict[str, Any]:
    """
    Convert a DocumentPayload (and all nested dataclasses) into a pure
    Python dictionary structure.

    Parameters
    ----------
    payload : DocumentPayload
        The document payload instance to convert.

    Returns
    -------
    Dict[str, Any]
        A dictionary representation of the payload, where all dataclass
        instances (DocumentPayload, SectionPayload, ParagraphPayload,
        SentencePayload) are recursively converted into dictionaries.
    """
    # asdict recursively walks all fields of the dataclass, including nested
    # dataclasses, lists, and dicts, producing a fully serializable structure.
    return asdict(payload)
