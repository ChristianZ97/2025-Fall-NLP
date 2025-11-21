# -*- coding: utf-8 -*-

# data_preprocess.py
# Dependency: pip install pylatexenc pandas requests

import csv
import os
import shutil
import tarfile
import requests
from pathlib import Path
from typing import List, Dict, Optional, Union
from urllib.parse import urlparse

import pandas as pd
from pylatexenc.latexwalker import (
    LatexWalker,
    LatexEnvironmentNode,
    LatexMacroNode,
    LatexGroupNode,
    LatexCharsNode,
)


class ArxivDownloader:
    """
    Handles downloading ArXiv data, including PDFs and Source (LaTeX) files.
    """

    def __init__(self, download_root: Union[str, Path]):
        self.root = Path(download_root)
        self.pdf_dir = self.root / "pdfs"
        self.source_dir = self.root / "sources"

        # Ensure directories exist
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)

    def load_metadata(self, csv_path: Path) -> List[Dict[str, str]]:
        """Reads metadata from a CSV file."""
        with csv_path.open(mode="r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def download_pdf(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """Downloads a single PDF."""
        if not filename:
            filename = Path(urlparse(url).path).name
            if not filename.endswith(".pdf"):
                filename += ".pdf"

        save_path = self.pdf_dir / filename

        if save_path.exists():
            print(f"[PDF] Exists, skipping: {filename}")
            return save_path

        try:
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
        Downloads and extracts the LaTeX source file (tar.gz).
        Returns the directory of the extracted files.
        """
        # Clean ID: remove version numbers or extra extensions
        clean_id = arxiv_id.split("/")[-1].replace(".pdf", "")
        # Remove version suffix (e.g., v1, v2) to get the latest version,
        # or keep it if you need a specific version.
        # Usually for source, just the base ID is safest: 2405.01814
        if "v" in clean_id and clean_id.split("v")[-1].isdigit():
            clean_id = clean_id.split("v")[0]

        # Correct Endpoint: /src/ gets the source tarball directly
        url = f"https://arxiv.org/src/{clean_id}"

        tar_path = self.source_dir / f"{clean_id}.tar.gz"
        extract_path = self.source_dir / clean_id

        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"[Source] Exists, skipping: {clean_id}")
            return extract_path

        print(f"[Source] Downloading: {clean_id} from {url}...")

        # Essential: ArXiv requires a User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()

            # Check content type to ensure we got a file, not an HTML error page
            content_type = response.headers.get("content-type", "")
            if "html" in content_type.lower():
                print(
                    f"[Source] Error: Retrieved HTML page instead of file for {clean_id}. Check URL or permissions."
                )
                return None

            with open(tar_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)

            extract_path.mkdir(exist_ok=True)

            # ArXiv sources are usually .tar.gz, but sometimes just .gz or .pdf
            try:
                if tarfile.is_tarfile(tar_path):
                    with tarfile.open(tar_path, "r:*") as tar:
                        tar.extractall(path=extract_path, filter="data")
                else:
                    print(f"[Source] Warning: {clean_id} is not a standard tar file.")
                    # Could add logic here to handle single file gz if needed
                    return None

                return extract_path
            except Exception as e:
                print(f"[Source] Extraction failed: {e}")
                return None
            finally:
                if tar_path.exists():
                    tar_path.unlink()

        except requests.RequestException as e:
            print(f"[Source] Download failed {clean_id}: {e}")
            return None


class TexProcessor:
    """
    Handles parsing of LaTeX file structure.
    """

    def __init__(self):
        pass

    def find_main_tex(self, source_dir: Path) -> Optional[Path]:
        """Attempts to find the main .tex file in the directory (usually the largest one)."""
        if not source_dir or not source_dir.exists():
            return None

        tex_files = list(source_dir.glob("*.tex"))
        if not tex_files:
            return None

        # Simple heuristic: Choose the largest file as main.tex
        return max(tex_files, key=lambda p: p.stat().st_size)

    def parse_file(self, tex_path: Path) -> List[Dict[str, str]]:
        """
        Parses a .tex file and returns a structured list of sections.
        """
        if not tex_path or not tex_path.exists():
            return []

        with open(tex_path, "r", encoding="utf-8", errors="replace") as f:
            latex_content = f.read()

        walker = LatexWalker(latex_content)
        try:
            nodes, _, _ = walker.get_latex_nodes()
        except Exception as e:
            print(f"[Tex] Parse error {tex_path.name}: {e}")
            return []

        structure = []
        current_section = "Abstract/Intro"

        # Internal recursive function to walk the node tree
        def _walk(node_list):
            nonlocal current_section
            text_buffer = []

            for node in node_list:
                if isinstance(node, LatexMacroNode):
                    if node.macroname in ["section", "subsection", "subsubsection"]:
                        # 1. Save previous buffer content before switching sections
                        if text_buffer:
                            clean_text = "".join(text_buffer).strip()
                            if clean_text:
                                structure.append(
                                    {"section": current_section, "text": clean_text}
                                )
                            text_buffer = []

                        # 2. Update section title
                        # Simplified logic; actual implementation might need to parse
                        # node.nodeargd to get the exact title text.
                        current_section = node.macroname

                    # Handle other Macros, e.g., \input{...}
                    # If recursive reading of other files is needed, extend here.

                elif isinstance(node, LatexEnvironmentNode):
                    _walk(node.nodelist)
                elif isinstance(node, LatexGroupNode):
                    _walk(node.nodelist)
                elif isinstance(node, LatexCharsNode):
                    text_buffer.append(node.chars)

            # Flush remaining buffer at the end of the loop
            if text_buffer:
                clean_text = "".join(text_buffer).strip()
                if clean_text:
                    structure.append({"section": current_section, "text": clean_text})

        _walk(nodes)
        return structure


def _flatten_sections_to_text(title: str, sections: list[dict]) -> str:
    """Helper to format the structured data into a clean string."""
    lines = [f"Document ID: {title}", "=" * 40, ""]

    for sec in sections:
        # Add Section Header
        header = sec.get("section", "Unknown Section")
        lines.append(f"\n# {header}")
        lines.append("-" * len(header))

        # Add Content
        text = sec.get("text", "").strip()
        if text:
            lines.append(text)

    return "\n".join(lines)


def run_processing_pipeline(
    df: pd.DataFrame,
    output_root: Union[str, Path],
    download_pdf: bool = True,
    download_source: bool = True,
):
    """
    Orchestrates the entire pipeline:
    1. Downloads PDF (optional).
    2. Downloads and extracts LaTeX source.
    3. Parses the main .tex file and saves it as a plain .txt file.

    Args:
        df: DataFrame containing 'id' and 'url' columns.
        output_root: Root directory for saving data.
        download_pdf: Whether to download the PDF file.
        download_source: Whether to download and process the source LaTeX.
    """
    root = Path(output_root)
    txt_dir = root / "texts"
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Initialize workers
    downloader = ArxivDownloader(download_root=root)
    processor = TexProcessor()

    print(f"Starting pipeline for {len(df)} documents...")

    for _, row in df.iterrows():
        arxiv_id = str(row.get("id", "")).strip()
        url = row.get("url", "")

        if not arxiv_id:
            continue

        print(f"\n=== Processing {arxiv_id} ===")

        # 1. Download PDF
        if download_pdf:
            downloader.download_pdf(url, filename=f"{arxiv_id}.pdf")

        # 2. Process Source to Text
        if download_source:
            # Download and extract tarball
            source_path = downloader.download_source(url)

            if not source_path:
                print(f"Skipping text conversion for {arxiv_id} (No source).")
                continue

            # Find main .tex file
            main_tex = processor.find_main_tex(source_path)
            if not main_tex:
                print(f"Skipping text conversion for {arxiv_id} (No .tex file found).")
                continue

            # Parse LaTeX structure
            parsed_sections = processor.parse_file(main_tex)

            # Convert to plain text string
            full_text = _flatten_sections_to_text(arxiv_id, parsed_sections)

            # Save to .txt
            txt_path = txt_dir / f"{arxiv_id}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            print(f"[Text] Saved processed text to: {txt_path.name}")

    print("\nPipeline completed.")


# --- Usage Example ---
if __name__ == "__main__":
    # 1. Setup Data Path
    data_path = "./data/WattBot2025"
    metadata_file = os.path.join(data_path, "metadata.csv")

    # 2. Check if metadata exists before running
    if os.path.exists(metadata_file):
        try:
            # Fix: Added encoding='latin1' to handle non-utf-8 characters
            df_meta = pd.read_csv(metadata_file, encoding="latin1")
        except UnicodeDecodeError:
            # Fallback: Try cp1252 if latin1 fails
            df_meta = pd.read_csv(metadata_file, encoding="cp1252")

        # 3. Run Pipeline
        run_processing_pipeline(
            df=df_meta,
            output_root=data_path + "/download",
            download_pdf=True,
            download_source=True,
        )
    else:
        print(f"Error: Metadata file not found at {metadata_file}")
