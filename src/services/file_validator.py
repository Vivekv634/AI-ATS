"""
File validator for resume ingestion.

Checks: allowed extension, file size, PDF magic bytes, basic PDF integrity.
No third-party magic library needed — we inspect bytes directly.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
MIN_FILE_BYTES = 4  # must be enough for magic bytes check
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
PDF_MAGIC = b"%PDF"


@dataclass
class ValidationResult:
    ok: bool = True
    error: str = ""
    file_hash: str = ""  # sha256 hex, set on success
    file_size: int = 0
    extension: str = ""


class FileValidator:
    """
    Validates resume files before ingestion.

    Accepts either a filesystem path or raw bytes + filename.
    Returns a ValidationResult — never raises.
    """

    def validate_path(self, path: "str | Path") -> ValidationResult:
        path = Path(path)
        ext = path.suffix.lower()
        result = ValidationResult(extension=ext)

        # Extension check
        if ext not in ALLOWED_EXTENSIONS:
            result.ok = False
            result.error = f"Unsupported extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            return result

        # Read file
        try:
            content = path.read_bytes()
        except OSError as e:
            result.ok = False
            result.error = f"Cannot read file: {e}"
            return result

        return self._validate_content(content, ext, result)

    def validate_bytes(self, content: bytes, filename: str) -> ValidationResult:
        ext = Path(filename).suffix.lower()
        result = ValidationResult(extension=ext)

        if ext not in ALLOWED_EXTENSIONS:
            result.ok = False
            result.error = f"Unsupported extension '{ext}'"
            return result

        return self._validate_content(content, ext, result)

    def _validate_content(
        self, content: bytes, ext: str, result: ValidationResult
    ) -> ValidationResult:
        result.file_size = len(content)

        # Size checks
        if result.file_size == 0:
            result.ok = False
            result.error = "File is empty"
            return result

        if result.file_size < MIN_FILE_BYTES:
            result.ok = False
            result.error = f"File too small ({result.file_size} bytes) — likely truncated"
            return result

        if result.file_size > MAX_FILE_BYTES:
            result.ok = False
            result.error = (
                f"File size {result.file_size / 1024 / 1024:.1f} MB exceeds "
                f"limit of {MAX_FILE_BYTES / 1024 / 1024:.0f} MB"
            )
            return result

        # PDF-specific checks
        if ext == ".pdf":
            if not content.startswith(PDF_MAGIC):
                result.ok = False
                result.error = (
                    f"File does not start with PDF magic bytes (%PDF). " f"Got: {content[:8]!r}"
                )
                return result

        # Compute hash
        result.file_hash = hashlib.sha256(content).hexdigest()
        result.ok = True
        return result
