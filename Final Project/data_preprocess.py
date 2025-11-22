# data_preprocess.py

import csv
import os
import shutil
import tarfile
import json
import requests

from pathlib import Path
from typing import List, Dict, Optional, Union, Any

from urllib.parse import urlparse, urlsplit

import pandas as pd

from pylatexenc.latexwalker import (
    LatexWalker,
    LatexEnvironmentNode,
    LatexMacroNode,
    LatexGroupNode,
    LatexCharsNode,
)

from pypdf import PdfReader
from pypdf.generic import DictionaryObject, IndirectObject


from types import (
    DocumentPayload,
    ParagraphPayload,
    SectionPayload,
    SentencePayload,
    payload_to_dict,
)


import re
import nltk

nltk.download("punkt")

from nltk.tokenize import sent_tokenize


def split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in sent_tokenize(text)]


# Global configuration defaults
MODE = "tex"  # "tex" = parse LaTeX source, "pdf" = parse PDF text
DOWNLOAD_PDF = True  # Whether to download PDFs
DOWNLOAD_SOURCE = True  # Whether to download LaTeX source tarballs
PDF_LIMIT = None  # If set to an int, stop after processing this many PDFs


class ArxivDownloader:
    """
    Handles downloading of PDF and LaTeX source files from arxiv.org
    and organizing them into local directories.

    Attributes
    ----------
    root : Path
        Root directory for all downloads.
    pdf_dir : Path
        Subdirectory where PDF files are stored.
    source_dir : Path
        Subdirectory where LaTeX source archives are stored.
    """

    def __init__(self, download_root: Union[str, Path]):
        # Normalize root path
        self.root = Path(download_root)
        # Create subdirectories for PDFs and sources if they do not exist
        self.pdf_dir = self.root / "pdfs"
        self.source_dir = self.root / "sources"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)

    def load_metadata(self, csv_path: Path) -> List[Dict[str, str]]:
        """
        Load metadata from a CSV file into a list of dicts.

        Each row becomes a dict with column names as keys.

        Parameters
        ----------
        csv_path : Path
            Path to the metadata CSV file.

        Returns
        -------
        List[Dict[str, str]]
            List of row dictionaries.
        """
        with csv_path.open(mode="r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def download_pdf(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """
        Download a PDF from a given URL, save it under pdf_dir, and return the local path.

        If filename is not provided, it is inferred from the URL and forced to end with '.pdf'.

        Parameters
        ----------
        url : str
            URL of the PDF to download.
        filename : Optional[str]
            Optional filename for saving the PDF.

        Returns
        -------
        Optional[Path]
            Local path to the saved PDF file, or None on failure.
        """
        if not filename:
            # Extract the file name from the URL path if not explicitly provided
            filename = Path(urlparse(url).path).name
            if not filename.endswith(".pdf"):
                filename += ".pdf"

        save_path = self.pdf_dir / filename

        # Skip download if file already exists
        if save_path.exists():
            print(f"[PDF] Exists, skipping: {filename}")
            return save_path

        try:
            # Stream download to avoid loading entire file in memory
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with save_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[PDF] Download successful: {filename}")
            return save_path
        except requests.RequestException as e:
            print(f"[PDF] Download failed {url}: {e}")
            return None

    def download_source(self, arxiv_id: str) -> Optional[Path]:
        """
        Download the LaTeX source archive for an arXiv paper and extract it.

        Notes
        -----
        - arxiv_id can be a full URL or just an ID; this method normalizes it.
        - The extracted files are placed in `source_dir/<clean_id>/`.
        - The .tar.gz archive is deleted after successful extraction.

        Parameters
        ----------
        arxiv_id : str
            Raw arXiv identifier or related string, possibly including version or .pdf.

        Returns
        -------
        Optional[Path]
            Path to the extracted source directory, or None if download or extraction fails.
        """
        # Normalize the arxiv_id:
        # - remove URL prefixes
        # - drop ".pdf" suffix
        clean_id = arxiv_id.split("/")[-1].replace(".pdf", "")
        # Remove version suffix like "v2" if present
        if "v" in clean_id and clean_id.split("v")[-1].isdigit():
            clean_id = clean_id.split("v")[0]

        # arXiv LaTeX source download URL pattern
        url = f"https://arxiv.org/src/{clean_id}"

        # Local paths: archive and extraction directory
        tar_path = self.source_dir / f"{clean_id}.tar.gz"
        extract_path = self.source_dir / clean_id

        # If already extracted and non-empty, skip re-download
        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"[Source] Exists, skipping: {clean_id}")
            return extract_path

        print(f"[Source] Downloading: {clean_id} from {url}...")

        # Use a common browser-like user agent to reduce blocks
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        try:
            # Stream the response directly to a local tar.gz file
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()

            # If server returns HTML, it usually indicates an error or permission issue
            content_type = response.headers.get("content-type", "")
            if "html" in content_type.lower():
                print(
                    f"[Source] Error: Retrieved HTML page instead of file for {clean_id}. "
                    f"Check URL or permissions."
                )
                return None

            # Save the raw response to tar_path
            with open(tar_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)

            # Ensure extraction directory exists
            extract_path.mkdir(exist_ok=True)

            try:
                # Validate if the file is a proper tar archive
                if tarfile.is_tarfile(tar_path):
                    with tarfile.open(tar_path, "r:*") as tar:
                        # 'filter="data"' avoids extracting special members (Python 3.12+)
                        tar.extractall(path=extract_path, filter="data")
                else:
                    print(f"[Source] Warning: {clean_id} is not a standard tar file.")
                    return None

                return extract_path
            except Exception as e:
                print(f"[Source] Extraction failed: {e}")
                return None
            finally:
                # Clean up archive file after extraction attempt
                if tar_path.exists():
                    tar_path.unlink()

        except requests.RequestException as e:
            print(f"[Source] Download failed {clean_id}: {e}")
            return None


class TexProcessor:
    """
    Process LaTeX source files into a structured DocumentPayload.

    Responsibilities
    ----------------
    - Find the main .tex file in a directory (heuristic: largest .tex file).
    - Parse LaTeX content into sections and associated text blocks.
    - Convert parsed text into sections, paragraphs, and sentences.
    """

    def __init__(self):
        # Currently no stateful attributes are needed
        pass

    def find_main_tex(self, source_dir: Path) -> Optional[Path]:
        """
        Heuristically determine the "main" LaTeX file in a source directory.

        Strategy
        --------
        - Look for all *.tex files directly under `source_dir`.
        - Select the largest file by size, assuming it is the main document file.

        Parameters
        ----------
        source_dir : Path
            Directory containing extracted LaTeX files.

        Returns
        -------
        Optional[Path]
            Path to the main .tex file, or None if none are found.
        """
        if not source_dir or not source_dir.exists():
            return None

        tex_files = list(source_dir.glob("*.tex"))
        if not tex_files:
            return None

        # Choose the .tex file with the largest size (in bytes)
        return max(tex_files, key=lambda p: p.stat().st_size)

    def parse_file(self, tex_path: Path) -> List[Dict[str, str]]:
        """
        Parse a LaTeX file into a list of sections with raw text.

        This uses pylatexenc's LatexWalker to traverse the LaTeX AST and extract
        text nodes, while tracking section / subsection commands as logical section
        boundaries.

        Output example
        --------------
        [
            {"section": "section", "text": "... full text of section ..."},
            {"section": "subsection", "text": "..."},
            ...
        ]

        Parameters
        ----------
        tex_path : Path
            Path to the .tex file to parse.

        Returns
        -------
        List[Dict[str, str]]
            A list of dicts each containing 'section' and 'text' keys.
        """
        if not tex_path or not tex_path.exists():
            return []

        # Read LaTeX content, tolerate encoding issues by replacing invalid chars
        with open(tex_path, "r", encoding="utf-8", errors="replace") as f:
            latex_content = f.read()

        walker = LatexWalker(latex_content)
        try:
            # nodes: top-level node list from the parsed LaTeX document
            nodes, _, _ = walker.get_latex_nodes()
        except Exception as e:
            print(f"[Tex] Parse error {tex_path.name}: {e}")
            return []

        structure = []
        # Default section label used before encountering any explicit \section macro
        current_section = "Abstract/Intro"

        def _walk(node_list):
            """
            Recursively walk LaTeX nodes and accumulate text according to sections.

            This inner function captures `current_section` and `structure` from
            the outer scope.
            """
            nonlocal current_section
            text_buffer = []

            for node in node_list:
                # Section-like macros: \section, \subsection, \subsubsection
                if isinstance(node, LatexMacroNode):
                    if node.macroname in ["section", "subsection", "subsubsection"]:
                        # Flush gathered text into structure under the previous section
                        if text_buffer:
                            clean_text = "".join(text_buffer).strip()
                            if clean_text:
                                structure.append(
                                    {"section": current_section, "text": clean_text}
                                )
                            text_buffer = []

                        # Switch current section label to the name of the macro
                        # (Note: does not capture the actual section title text)
                        current_section = node.macroname

                # Environments and groups can contain nested nodes; recurse into them
                elif isinstance(node, LatexEnvironmentNode):
                    _walk(node.nodelist)
                elif isinstance(node, LatexGroupNode):
                    _walk(node.nodelist)

                # Plain text characters: append to the current buffer
                elif isinstance(node, LatexCharsNode):
                    text_buffer.append(node.chars)

            # After iterating, flush any remaining text as a final block
            if text_buffer:
                clean_text = "".join(text_buffer).strip()
                if clean_text:
                    structure.append({"section": current_section, "text": clean_text})

        # Start traversal from the top-level nodes
        _walk(nodes)
        return structure

    def to_payload(
        self,
        tex_path: Path,
        *,
        doc_id: str,
        title: str,
        metadata: dict[str, Any],
    ) -> DocumentPayload:
        """
        Convert a LaTeX .tex file into a DocumentPayload.

        Steps
        -----
        1. Parse the LaTeX file into "raw_sections" via parse_file.
        2. Split each section text into paragraphs.
        3. Split each paragraph into sentences.
        4. Aggregate everything into a DocumentPayload structure.

        Parameters
        ----------
        tex_path : Path
            Path to the main .tex file.
        doc_id : str
            Unique identifier for the document.
        title : str
            Human-readable title of the document.
        metadata : dict[str, Any]
            Arbitrary additional metadata to attach to the DocumentPayload.

        Returns
        -------
        DocumentPayload
            Structured representation of the document content.
        """
        raw_sections = self.parse_file(tex_path)
        sections: list[SectionPayload] = []
        all_paragraph_texts: list[str] = []

        for sec in raw_sections:
            # Use section macro name or fallback label
            sec_title = sec.get("section") or "Unknown Section"
            sec_text = sec.get("text", "") or ""
            paragraphs: list[ParagraphPayload] = []

            # Split each section text into paragraphs
            for paragraph_text in split_paragraphs(sec_text):
                # Split paragraphs into sentences
                sentences = [
                    SentencePayload(text=s) for s in split_sentences(paragraph_text)
                ]
                # Build paragraph payload object
                paragraphs.append(
                    ParagraphPayload(
                        text=paragraph_text,
                        sentences=sentences or None,
                        metadata={},
                    )
                )
                all_paragraph_texts.append(paragraph_text)

            # Build section payload if there are any paragraphs
            if paragraphs:
                sections.append(
                    SectionPayload(
                        title=sec_title,
                        paragraphs=paragraphs,
                        metadata={},
                    )
                )

        # Concatenate all paragraph strings for the top-level "text" field
        combined_text = "\n\n".join(all_paragraph_texts)
        return DocumentPayload(
            document_id=doc_id,
            title=title,
            text=combined_text,
            metadata=metadata,
            sections=sections,
        )


class PdfProcessor:
    """
    Process PDF files into a structured DocumentPayload.

    Responsibilities
    ----------------
    - Detect if a URL points to a valid PDF resource.
    - Extract raw text for each page of the PDF.
    - Detect images on each page (basic metadata only).
    - Build sections per page, containing paragraphs and sentences.

    Note
    ----
    - Text extraction quality depends heavily on the PDF structure.
    - Image extraction here only captures dimensions and color space,
      not the image binary content.
    """

    def __init__(self):
        # Currently no stateful attributes are needed
        pass

    @staticmethod
    def clean_url(url: str) -> str:
        """
        Trim whitespace from URL; return empty string if None.
        """
        return (url or "").strip()

    @staticmethod
    def is_pdf_url(url: str) -> bool:
        """
        Heuristic to check if a URL looks like it is pointing to a PDF file.

        Rules
        -----
        - If the final path of the URL ends with '.pdf' (case-insensitive).
        - Special handling for arxiv.org: path starting with '/pdf/'.

        Parameters
        ----------
        url : str
            URL string to test.

        Returns
        -------
        bool
            True if the URL looks like a PDF URL, False otherwise.
        """
        cleaned = PdfProcessor.clean_url(url)
        if not cleaned:
            return False
        parts = urlsplit(cleaned)
        base = f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")
        path_lower = parts.path.lower().rstrip("/")
        # Standard ".pdf" suffix
        if base.lower().endswith(".pdf"):
            return True
        # arxiv.org special case, e.g. https://arxiv.org/pdf/1234.5678.pdf
        if parts.netloc.endswith("arxiv.org") and path_lower.startswith("/pdf/"):
            return True
        return False

    @staticmethod
    def has_pdf_mime(url: str) -> bool:
        """
        Send a HEAD request to check if the server advertises 'pdf' in Content-Type.

        This is a fallback when the URL does not clearly end with '.pdf'.

        Parameters
        ----------
        url : str
            URL to query via HTTP HEAD.

        Returns
        -------
        bool
            True if server returns a Content-Type that includes 'pdf', else False.
        """
        try:
            resp = requests.head(url, allow_redirects=True, timeout=30)
            ctype = resp.headers.get("Content-Type", "").lower()
            return "pdf" in ctype
        except requests.RequestException:
            return False

    @staticmethod
    def _resolve(obj):
        """
        Resolve an IndirectObject to its underlying object, if needed.

        Parameters
        ----------
        obj : Any
            pypdf object, possibly an IndirectObject.

        Returns
        -------
        Any
            Resolved object or the original if already direct.
        """
        return obj.get_object() if isinstance(obj, IndirectObject) else obj

    @staticmethod
    def _extract_images(page) -> list[dict[str, Any]]:
        """
        Inspect a PDF page and extract basic metadata for embedded images.

        Extraction details
        ------------------
        - Looks up the /Resources and /XObject dictionaries.
        - For each XObject with /Subtype == /Image, records:
          * name
          * width
          * height
          * color space

        Parameters
        ----------
        page : pypdf._page.PageObject
            A single PDF page object.

        Returns
        -------
        list[dict[str, Any]]
            List of image metadata dictionaries.
        """
        images: list[dict[str, Any]] = []
        resources = page.get("/Resources")
        if not resources:
            return images
        resources = PdfProcessor._resolve(resources)

        # XObject dictionary stores various graphical objects including images
        xobject = (
            resources.get("/XObject")
            if isinstance(resources, DictionaryObject)
            else None
        )
        if xobject is None:
            return images
        xobject = PdfProcessor._resolve(xobject)
        if not isinstance(xobject, DictionaryObject):
            return images

        for name, obj in xobject.items():
            resolved = PdfProcessor._resolve(obj)
            if not isinstance(resolved, DictionaryObject):
                continue
            subtype = resolved.get("/Subtype")
            # Only interested in image subtypes
            if subtype == "/Image":
                images.append(
                    {
                        "name": str(name),
                        "width": resolved.get("/Width"),
                        "height": resolved.get("/Height"),
                        "color_space": resolved.get("/ColorSpace"),
                    }
                )
        return images

    def to_payload(
        self,
        pdf_path: Path,
        *,
        doc_id: str,
        title: str,
        metadata: dict[str, Any],
    ) -> DocumentPayload:
        """
        Convert a PDF file into a DocumentPayload.

        Steps
        -----
        1. Load the PDF using pypdf.
        2. For each page:
           a. Extract raw text.
           b. Split into paragraphs and sentences.
           c. Detect images, and add placeholder paragraphs for them.
        3. Concatenate all paragraph texts into the top-level `text` field.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file.
        doc_id : str
            Unique document identifier.
        title : str
            Human-readable title.
        metadata : dict[str, Any]
            Additional fields to attach to the DocumentPayload.

        Returns
        -------
        DocumentPayload
            Structured representation of the PDF content.
        """
        reader = PdfReader(str(pdf_path))
        sections: list[SectionPayload] = []
        all_paragraph_texts: list[str] = []

        # Iterate over each page in the PDF
        for page_num, page in enumerate(reader.pages, start=1):
            # Extract raw text for the page (may be empty if PDF is image-based)
            raw_text = page.extract_text() or ""
            paragraphs: list[ParagraphPayload] = []

            # Split page text into paragraphs and sentences
            for paragraph_text in split_paragraphs(raw_text):
                sentences = [
                    SentencePayload(text=sentence)
                    for sentence in split_sentences(paragraph_text)
                ]
                paragraphs.append(
                    ParagraphPayload(
                        text=paragraph_text,
                        sentences=sentences or None,
                        metadata={"page": page_num},
                    )
                )
                all_paragraph_texts.append(paragraph_text)

            # Extract image metadata and add a textual placeholder paragraph for each
            images = self._extract_images(page)
            for idx, info in enumerate(images, start=1):
                caption = (
                    f"[Image page={page_num} idx={idx}] Placeholder for "
                    f"{info.get('width')}x{info.get('height')} "
                    f"{info.get('color_space')} graphic."
                )
                sentences = [SentencePayload(text=caption)]
                paragraphs.append(
                    ParagraphPayload(
                        text=caption,
                        sentences=sentences,
                        metadata={
                            "page": page_num,
                            "image_index": idx,
                            "placeholder": True,
                        },
                    )
                )
                all_paragraph_texts.append(caption)

            # Wrap this page's paragraphs in a section named "Page X"
            if paragraphs:
                sections.append(
                    SectionPayload(
                        title=f"Page {page_num}",
                        paragraphs=paragraphs,
                        metadata={"page": page_num},
                    )
                )

        # Join all paragraphs for a global plain-text view
        combined_text = "\n\n".join(all_paragraph_texts)
        return DocumentPayload(
            document_id=doc_id,
            title=title,
            text=combined_text,
            metadata=metadata,
            sections=sections,
        )

    def to_markdown(
        self,
        pdf_path: Path,
        *,
        doc_id: str,
        title: str,
        metadata: dict[str, Any],
    ) -> str:
        """
        Convenience method: convert a PDF directly into Markdown string.

        Each page becomes a "## Page X" section heading.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file.
        doc_id : str
            Document identifier.
        title : str
            Title for the top-level "# ..." heading.
        metadata : dict[str, Any]
            Metadata to pass through to the intermediate payload (unused here).

        Returns
        -------
        str
            Markdown-formatted text representation of the PDF.
        """
        payload = self.to_payload(
            pdf_path, doc_id=doc_id, title=title, metadata=metadata
        )
        lines = [f"# {title}", ""]
        for section in payload.sections or []:
            lines.append(f"## {section.title}")
            lines.append("")
            for paragraph in section.paragraphs:
                lines.append(paragraph.text)
                lines.append("")
        return "\n".join(lines)


def run_processing_pipeline(
    df: pd.DataFrame,
    output_root: Union[str, Path],
    download_pdf: bool = True,
    download_source: bool = True,
    mode: str = "tex",
    pdf_limit: Optional[int] = None,
):
    """
    High-level orchestration for processing a batch of documents.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata dataframe. Expected columns:
        - 'id': unique document ID
        - 'url': PDF/LaTeX URL (often arXiv)
        - 'title': document title (optional)
        - 'type', 'year': optional metadata fields
    output_root : Union[str, Path]
        Root directory for outputs (JSON, TXT, downloads).
    download_pdf : bool, default True
        If True, download PDFs from 'url' column.
    download_source : bool, default True
        If True and mode == 'tex', download LaTeX source.
    mode : str, default "tex"
        Either "tex" (use LaTeX source) or "pdf" (use PDF text).
    pdf_limit : Optional[int], default None
        If set and mode == "pdf", stop after processing this many PDFs.

    Behavior
    --------
    - Creates subdirectories under output_root:
        * 'docs'  : JSON payloads
        * 'texts' : plain text files
        * 'pdfs'  : downloaded PDFs (via ArxivDownloader)
        * 'sources': downloaded LaTeX sources (via ArxivDownloader)
    - For each row in df, processes according to the chosen mode.
    """
    mode = mode.lower()
    if mode not in {"tex", "pdf"}:
        raise ValueError(f"Unknown mode: {mode} (expected 'tex' or 'pdf')")

    root = Path(output_root)
    docs_dir = root / "docs"
    txt_dir = root / "texts"
    docs_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    downloader = ArxivDownloader(download_root=root)
    tex_processor = TexProcessor()
    pdf_processor = PdfProcessor()

    print(f"Starting pipeline for {len(df)} documents in mode='{mode}'...")

    processed = 0  # Track how many documents have been successfully processed

    # Iterate over each row (document) in the metadata dataframe
    for _, row in df.iterrows():
        # Read basic metadata fields
        doc_id = str(row.get("id", "")).strip()
        url = row.get("url", "") or ""
        if not doc_id:
            # Skip rows without a valid ID
            continue

        print(f"\n=== Processing {doc_id} ===")

        title = row.get("title") or doc_id
        meta = {
            "url": url,
            "type": row.get("type"),
            "year": row.get("year"),
        }

        # --------------------
        # PDF MODE PROCESSING
        # --------------------
        if mode == "pdf":
            # First, ensure the URL looks like a PDF or has a PDF MIME type
            if not pdf_processor.is_pdf_url(url) and not pdf_processor.has_pdf_mime(
                url
            ):
                print(f"[skip] {doc_id}: URL does not look like PDF: {url}")
                continue

            pdf_path: Optional[Path] = None
            if download_pdf:
                # Download PDF to local storage
                pdf_path = downloader.download_pdf(url, filename=f"{doc_id}.pdf")
            else:
                # If not downloading, expect that a corresponding local PDF already exists
                candidate = downloader.pdf_dir / f"{doc_id}.pdf"
                if candidate.exists():
                    pdf_path = candidate

            if not pdf_path or not pdf_path.exists():
                print(f"[skip] {doc_id}: PDF file not found, skip.")
                continue

            try:
                # Convert PDF to DocumentPayload
                payload = pdf_processor.to_payload(
                    pdf_path,
                    doc_id=doc_id,
                    title=title,
                    metadata=meta,
                )
            except Exception as exc:
                print(f"[error] {doc_id}: PDF conversion failed: {exc}")
                continue

            # Save JSON representation
            json_path = docs_dir / f"{doc_id}.json"
            json_path.write_text(
                json.dumps(payload_to_dict(payload), ensure_ascii=False),
                encoding="utf-8",
            )
            # Save plain text
            txt_path = txt_dir / f"{doc_id}.txt"
            txt_path.write_text(payload.text, encoding="utf-8")

            print(f"[ok] {doc_id} (PDF) -> {json_path} & {txt_path}")
            processed += 1

            # Stop early if pdf_limit is reached
            if pdf_limit is not None and processed >= pdf_limit:
                print(f"Reached pdf_limit={pdf_limit}, stop.")
                break

            # Continue to next document
            continue

        # ---------------------
        # TEX MODE PROCESSING
        # ---------------------

        # Optionally download the PDF even when in tex mode (for reference)
        if download_pdf:
            downloader.download_pdf(url, filename=f"{doc_id}.pdf")

        # If source downloading is disabled, skip LaTeX processing for this doc
        if not download_source:
            print(f"[skip] {doc_id}: download_source=False, skip tex parsing.")
            continue

        # Download LaTeX source. Note: function expects an arXiv-like identifier.
        source_path = downloader.download_source(url)
        if not source_path:
            print(f"Skipping tex conversion for {doc_id} (No source).")
            continue

        # Choose the main .tex file to parse
        main_tex = tex_processor.find_main_tex(source_path)
        if not main_tex:
            print(f"Skipping tex conversion for {doc_id} (No .tex file found).")
            continue

        try:
            # Convert LaTeX to DocumentPayload
            payload = tex_processor.to_payload(
                main_tex,
                doc_id=doc_id,
                title=title,
                metadata=meta,
            )
        except Exception as exc:
            print(f"[error] {doc_id}: Tex conversion failed: {exc}")
            continue

        # Save JSON representation
        json_path = docs_dir / f"{doc_id}.json"
        json_path.write_text(
            json.dumps(payload_to_dict(payload), ensure_ascii=False),
            encoding="utf-8",
        )
        # Save plain text
        txt_path = txt_dir / f"{doc_id}.txt"
        txt_path.write_text(payload.text, encoding="utf-8")

        print(f"[ok] {doc_id} (Tex) -> {json_path} & {txt_path}")

    print(f"\nPipeline completed. Processed {processed} documents.")


if __name__ == "__main__":
    """
    Entry point when running this script directly.

    Expects:
    - A directory './data/WattBot2025'
    - A metadata CSV file './data/WattBot2025/metadata.csv'
      with at least columns: 'id', 'url', 'title' (and optionally 'type', 'year').

    Behavior:
    - Reads metadata CSV with fallback encodings (latin1 then cp1252).
    - Invokes run_processing_pipeline with global configuration flags:
      MODE, DOWNLOAD_PDF, DOWNLOAD_SOURCE, PDF_LIMIT.
    """
    data_path = "./data/WattBot2025"
    metadata_file = os.path.join(data_path, "metadata.csv")

    if os.path.exists(metadata_file):
        try:
            # Try to read CSV assuming 'latin1' encoding first
            df_meta = pd.read_csv(metadata_file, encoding="latin1")
        except UnicodeDecodeError:
            # Fallback encoding in case of decoding errors
            df_meta = pd.read_csv(metadata_file, encoding="cp1252")

        run_processing_pipeline(
            df=df_meta,
            output_root=os.path.join(data_path, "download"),
            download_pdf=DOWNLOAD_PDF,
            download_source=DOWNLOAD_SOURCE,
            mode=MODE,
            pdf_limit=PDF_LIMIT,
        )
    else:
        print(f"Error: Metadata file not found at {metadata_file}")
