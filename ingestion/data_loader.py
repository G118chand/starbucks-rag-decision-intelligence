"""Document ingestion pipeline for structured CSV and unstructured Markdown sources."""

import logging
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Loads structured CSV reports and unstructured Markdown files into LangChain Documents."""

    def load_csv_files(self, directory: str) -> List[Document]:
        """Scan a directory for CSV files and convert each row to a LangChain Document.

        Args:
            directory: Path to the directory containing CSV files.

        Returns:
            List of Documents, one per non-empty row across all CSV files.
        """
        docs: List[Document] = []
        dir_path = Path(directory)

        csv_files = sorted(dir_path.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", directory)
            return docs

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.error("Failed to read %s: %s", csv_path, exc)
                continue

            filename = csv_path.name
            doc_type = csv_path.stem
            row_count = 0

            for _, row in df.iterrows():
                non_null = {col: row[col] for col in df.columns if pd.notna(row[col])}
                if not non_null:
                    continue

                page_content = f"[{filename}] " + " | ".join(
                    f"{col}: {val}" for col, val in non_null.items()
                )

                metadata: dict = {
                    "source": filename,
                    "data_type": "structured",
                    "document_type": doc_type,
                }
                if "region" in row and pd.notna(row["region"]):
                    metadata["region"] = row["region"]
                if "date" in row and pd.notna(row["date"]):
                    metadata["date"] = str(row["date"])

                docs.append(Document(page_content=page_content, metadata=metadata))
                row_count += 1

            logger.info("Loaded %d rows from %s", row_count, filename)

        return docs

    def load_markdown_files(self, directory: str) -> List[Document]:
        """Scan a directory for Markdown files and load each as a single Document.

        Args:
            directory: Path to the directory containing Markdown files.

        Returns:
            List of Documents, one per Markdown file.
        """
        docs: List[Document] = []
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning("Markdown directory does not exist: %s", directory)
            return docs

        md_files = sorted(dir_path.glob("*.md"))
        if not md_files:
            logger.warning("No Markdown files found in %s", directory)
            return docs

        for md_path in md_files:
            try:
                text = md_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.error("Failed to read %s: %s", md_path, exc)
                continue

            metadata: dict = {
                "source": md_path.name,
                "data_type": "unstructured",
                "document_type": "report",
            }
            docs.append(Document(page_content=text, metadata=metadata))
            logger.info("Loaded markdown file: %s", md_path.name)

        return docs

    def load_all(self) -> List[Document]:
        """Load all CSV and Markdown documents from the canonical data directories.

        Returns:
            Combined list of structured and unstructured Documents.
        """
        base_dir = Path(__file__).parent.parent / "data" / "synthetic"
        reports_dir = base_dir / "reports"

        structured_docs = self.load_csv_files(str(base_dir))
        unstructured_docs = self.load_markdown_files(str(reports_dir))

        all_docs = structured_docs + unstructured_docs
        logger.info(
            "Loaded %d documents (%d structured, %d unstructured)",
            len(all_docs),
            len(structured_docs),
            len(unstructured_docs),
        )
        return all_docs
