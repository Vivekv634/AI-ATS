"""
Base extractor class for document text extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExtractionResult:
    """Result of text extraction from a document."""

    text: str
    page_count: int = 1
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None

    @property
    def word_count(self) -> int:
        """Count words in extracted text."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Count characters in extracted text."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """Check if extraction resulted in empty text."""
        return len(self.text.strip()) == 0


class BaseExtractor(ABC):
    """
    Abstract base class for document text extractors.

    All format-specific extractors should inherit from this class.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Return tuple of supported file extensions (e.g., '.pdf', '.docx')."""
        pass

    def can_extract(self, file_path: str | Path) -> bool:
        """Check if this extractor can handle the given file."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def extract(self, file_path: str | Path) -> ExtractionResult:
        """
        Extract text content from a document.

        Args:
            file_path: Path to the document file

        Returns:
            ExtractionResult containing the extracted text and metadata
        """
        pass

    @abstractmethod
    def extract_from_bytes(
        self, content: bytes, filename: str = "document"
    ) -> ExtractionResult:
        """
        Extract text content from document bytes.

        Args:
            content: Raw bytes of the document
            filename: Original filename (for extension detection)

        Returns:
            ExtractionResult containing the extracted text and metadata
        """
        pass

    def _validate_file(self, file_path: str | Path) -> Path:
        """Validate that file exists and is readable."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        return path

    def _create_error_result(self, error: Exception) -> ExtractionResult:
        """Create an error result from an exception."""
        return ExtractionResult(
            text="",
            success=False,
            error_message=str(error),
        )
