"""
Import Center Dialog for AI-ATS.

Provides a unified interface for importing resumes from various sources:
- Local files and folders (with drag & drop)
- Google Drive (including Google Forms uploads)
- Google Sheets (for candidate metadata)
- Watch folders (auto-import)
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QFrame,
    QScrollArea,
    QGridLayout,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QMimeData
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent, QIcon

from src.utils.constants import COLORS, SUPPORTED_RESUME_FORMATS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImportTask:
    """Represents an import task."""
    source: str  # "local", "gdrive", "gsheets"
    file_path: Optional[str] = None
    file_name: str = ""
    status: str = "pending"  # pending, processing, success, error
    error_message: str = ""
    candidate_name: str = ""
    candidate_email: str = ""


class ImportWorker(QThread):
    """Background worker for importing resumes."""

    progress = pyqtSignal(int, int, str)  # current, total, message
    file_processed = pyqtSignal(dict)  # result dict
    finished = pyqtSignal(int, int)  # success_count, error_count

    def __init__(self, file_paths: list[str], parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """Process files in background."""
        success_count = 0
        error_count = 0

        try:
            from src.ml.nlp import get_resume_parser
            parser = get_resume_parser()
        except ImportError:
            self.finished.emit(0, len(self.file_paths))
            return

        for i, file_path in enumerate(self.file_paths):
            if self._cancelled:
                break

            self.progress.emit(i + 1, len(self.file_paths), Path(file_path).name)

            try:
                result = parser.parse_file(file_path)

                if result.success and result.contact:
                    candidate_data = {
                        "success": True,
                        "file_path": file_path,
                        "file_name": Path(file_path).name,
                        "first_name": result.contact.get("first_name") or "Unknown",
                        "last_name": result.contact.get("last_name") or "Candidate",
                        "email": result.contact.get("email") or "",
                        "phone": result.contact.get("phone") or "",
                        "skills": [s.get("name", "") for s in result.skills[:10]],
                        "experience_years": result.total_experience_years,
                        "education": result.education[0].get("degree") if result.education else "",
                    }
                    self.file_processed.emit(candidate_data)
                    success_count += 1
                else:
                    self.file_processed.emit({
                        "success": False,
                        "file_path": file_path,
                        "file_name": Path(file_path).name,
                        "error": "Parsing failed - insufficient data extracted",
                    })
                    error_count += 1

            except Exception as e:
                self.file_processed.emit({
                    "success": False,
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "error": str(e)[:100],
                })
                error_count += 1

        self.finished.emit(success_count, error_count)


class DropZone(QFrame):
    """Drag and drop zone for files."""

    files_dropped = pyqtSignal(list)  # List of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        self.setMinimumHeight(200)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
            }}
            QFrame:hover {{
                border-color: {COLORS['primary']};
                background-color: #f0f7ff;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Icon
        icon_label = QLabel("📁")
        icon_label.setFont(QFont("Segoe UI", 48))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        # Text
        text_label = QLabel("Drag & Drop Resume Files Here")
        text_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        text_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text_label)

        # Subtext
        subtext = QLabel("or click Browse to select files")
        subtext.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        subtext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtext)

        # Supported formats
        formats = QLabel(f"Supported: {', '.join(SUPPORTED_RESUME_FORMATS)}")
        formats.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; margin-top: 10px;")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(formats)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #e0f2fe;
                    border: 2px dashed {COLORS['primary']};
                    border-radius: 12px;
                }}
            """)

    def dragLeaveEvent(self, event):
        self._setup_ui()

    def dropEvent(self, event: QDropEvent):
        self._setup_ui()
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                if Path(path).suffix.lower() in SUPPORTED_RESUME_FORMATS:
                    files.append(path)
            elif os.path.isdir(path):
                # Recursively find resume files in folder
                for ext in SUPPORTED_RESUME_FORMATS:
                    files.extend([str(p) for p in Path(path).rglob(f"*{ext}")])

        if files:
            self.files_dropped.emit(files)
        else:
            QMessageBox.warning(
                self,
                "No Valid Files",
                "No supported resume files found.\n\n"
                f"Supported formats: {', '.join(SUPPORTED_RESUME_FORMATS)}"
            )


class LocalImportTab(QWidget):
    """Tab for importing from local files/folders."""

    import_requested = pyqtSignal(list)  # List of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        # Buttons row
        btn_layout = QHBoxLayout()

        browse_files_btn = QPushButton("📄 Browse Files")
        browse_files_btn.setMinimumHeight(40)
        browse_files_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_files_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #1d4ed8;
            }}
        """)
        browse_files_btn.clicked.connect(self._browse_files)
        btn_layout.addWidget(browse_files_btn)

        browse_folder_btn = QPushButton("📁 Browse Folder")
        browse_folder_btn.setMinimumHeight(40)
        browse_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_folder_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #16a34a;
            }}
        """)
        browse_folder_btn.clicked.connect(self._browse_folder)
        btn_layout.addWidget(browse_folder_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Selected files list
        self.files_list = QListWidget()
        self.files_list.setMinimumHeight(150)
        self.files_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['surface']};
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid #f1f5f9;
            }}
            QListWidget::item:selected {{
                background-color: #e0e7ff;
            }}
        """)
        layout.addWidget(QLabel("Selected Files:"))
        layout.addWidget(self.files_list)

        # Action buttons
        action_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet(f"color: {COLORS['text_secondary']};")
        clear_btn.clicked.connect(self._clear_files)
        action_layout.addWidget(clear_btn)

        action_layout.addStretch()

        self.import_btn = QPushButton("Import Selected Files")
        self.import_btn.setMinimumHeight(40)
        self.import_btn.setEnabled(False)
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1d4ed8;
            }}
            QPushButton:disabled {{
                background-color: #cbd5e1;
            }}
        """)
        self.import_btn.clicked.connect(self._import_files)
        action_layout.addWidget(self.import_btn)

        layout.addLayout(action_layout)

        self._selected_files = []

    def _browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Resume Files",
            "",
            "Documents (*.pdf *.docx *.doc *.txt);;All Files (*)",
        )
        if files:
            self._add_files(files)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Resumes",
        )
        if folder:
            files = []
            for ext in SUPPORTED_RESUME_FORMATS:
                files.extend([str(p) for p in Path(folder).rglob(f"*{ext}")])
            if files:
                self._add_files(files)
            else:
                QMessageBox.information(
                    self,
                    "No Files Found",
                    f"No resume files found in the selected folder.\n\n"
                    f"Supported formats: {', '.join(SUPPORTED_RESUME_FORMATS)}"
                )

    def _on_files_dropped(self, files: list):
        self._add_files(files)

    def _add_files(self, files: list):
        for file_path in files:
            if file_path not in self._selected_files:
                self._selected_files.append(file_path)
                item = QListWidgetItem(f"📄 {Path(file_path).name}")
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.files_list.addItem(item)

        self.import_btn.setEnabled(len(self._selected_files) > 0)
        self.import_btn.setText(f"Import {len(self._selected_files)} File(s)")

    def _clear_files(self):
        self._selected_files.clear()
        self.files_list.clear()
        self.import_btn.setEnabled(False)
        self.import_btn.setText("Import Selected Files")

    def _import_files(self):
        if self._selected_files:
            self.import_requested.emit(self._selected_files.copy())


class GoogleDriveTab(QWidget):
    """Tab for importing from Google Drive."""

    import_requested = pyqtSignal(list)  # List of file paths (downloaded)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._service = None
        self._current_folder_id = "root"
        self._folder_history = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Connection status
        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #fef3c7;
                border: 1px solid #fcd34d;
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        status_layout = QHBoxLayout(self.status_frame)

        self.status_label = QLabel("⚠️ Not connected to Google Drive")
        self.status_label.setStyleSheet("color: #92400e; font-weight: 500;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.connect_btn = QPushButton("Connect to Google Drive")
        self.connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.connect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #3367d6;
            }}
        """)
        self.connect_btn.clicked.connect(self._connect_gdrive)
        status_layout.addWidget(self.connect_btn)

        layout.addWidget(self.status_frame)

        # Search for Google Form
        search_group = QGroupBox("Quick Find: Google Form Responses")
        search_layout = QHBoxLayout(search_group)

        self.form_name_input = QLineEdit()
        self.form_name_input.setPlaceholderText("Enter Google Form name (e.g., 'Placement Registration 2024')")
        self.form_name_input.setMinimumHeight(36)
        search_layout.addWidget(self.form_name_input)

        self.find_form_btn = QPushButton("Find Form Folder")
        self.find_form_btn.setEnabled(False)
        self.find_form_btn.clicked.connect(self._find_form_folder)
        search_layout.addWidget(self.find_form_btn)

        layout.addWidget(search_group)

        # Folder browser
        browser_label = QLabel("Browse Google Drive:")
        browser_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(browser_label)

        # Navigation
        nav_layout = QHBoxLayout()

        self.back_btn = QPushButton("← Back")
        self.back_btn.setEnabled(False)
        self.back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self.back_btn)

        self.path_label = QLabel("/ My Drive")
        self.path_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        nav_layout.addWidget(self.path_label)

        nav_layout.addStretch()

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self._refresh_folder)
        nav_layout.addWidget(self.refresh_btn)

        layout.addLayout(nav_layout)

        # Splitter for folders and files
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Folders tree
        folders_widget = QWidget()
        folders_layout = QVBoxLayout(folders_widget)
        folders_layout.setContentsMargins(0, 0, 0, 0)

        folders_layout.addWidget(QLabel("Folders:"))
        self.folders_tree = QTreeWidget()
        self.folders_tree.setHeaderHidden(True)
        self.folders_tree.setMinimumWidth(250)
        self.folders_tree.itemDoubleClicked.connect(self._on_folder_double_clicked)
        self.folders_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {COLORS['surface']};
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }}
        """)
        folders_layout.addWidget(self.folders_tree)

        splitter.addWidget(folders_widget)

        # Files list
        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        files_layout.setContentsMargins(0, 0, 0, 0)

        files_layout.addWidget(QLabel("Resume Files:"))
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(3)
        self.files_table.setHorizontalHeaderLabels(["Name", "Type", "Size"])
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.files_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['surface']};
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }}
        """)
        files_layout.addWidget(self.files_table)

        splitter.addWidget(files_widget)
        splitter.setSizes([300, 500])

        layout.addWidget(splitter)

        # Import buttons
        import_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setEnabled(False)
        self.select_all_btn.clicked.connect(self._select_all_files)
        import_layout.addWidget(self.select_all_btn)

        import_layout.addStretch()

        self.import_gdrive_btn = QPushButton("Download & Import Selected")
        self.import_gdrive_btn.setMinimumHeight(40)
        self.import_gdrive_btn.setEnabled(False)
        self.import_gdrive_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1d4ed8;
            }}
            QPushButton:disabled {{
                background-color: #cbd5e1;
            }}
        """)
        self.import_gdrive_btn.clicked.connect(self._import_selected)
        import_layout.addWidget(self.import_gdrive_btn)

        layout.addLayout(import_layout)

        self._gdrive_files = []

    def _connect_gdrive(self):
        """Connect to Google Drive."""
        try:
            from src.services.google_drive_service import get_drive_service
            self._service = get_drive_service()

            if not self._service.is_available():
                QMessageBox.warning(
                    self,
                    "Missing Dependencies",
                    "Google API libraries not installed.\n\n"
                    "Install with:\n"
                    "pip install google-auth-oauthlib google-api-python-client"
                )
                return

            if not self._service.has_credentials():
                QMessageBox.warning(
                    self,
                    "Missing Credentials",
                    "Google Drive credentials not found.\n\n"
                    "Setup Instructions:\n"
                    "1. Go to console.cloud.google.com\n"
                    "2. Create project & enable Drive API\n"
                    "3. Create OAuth credentials\n"
                    "4. Download as credentials.json"
                )
                return

            # Try to authenticate
            self.status_label.setText("🔄 Connecting...")
            self.connect_btn.setEnabled(False)

            if self._service.authenticate():
                self._on_connected()
            else:
                self.status_label.setText("❌ Connection failed")
                self.connect_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Connection failed: {e}")
            self.connect_btn.setEnabled(True)

    def _on_connected(self):
        """Handle successful connection."""
        self.status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #d1fae5;
                border: 1px solid #6ee7b7;
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        self.status_label.setText("✓ Connected to Google Drive")
        self.status_label.setStyleSheet("color: #065f46; font-weight: 500;")
        self.connect_btn.setText("Reconnect")
        self.connect_btn.setEnabled(True)

        self.find_form_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.select_all_btn.setEnabled(True)

        self._refresh_folder()

    def _refresh_folder(self):
        """Refresh current folder contents."""
        if not self._service or not self._service.is_authenticated():
            return

        # Load folders
        self.folders_tree.clear()
        folders = self._service.list_folders(self._current_folder_id)

        for folder in folders:
            item = QTreeWidgetItem([f"📁 {folder.name}"])
            item.setData(0, Qt.ItemDataRole.UserRole, folder.id)
            self.folders_tree.addTopLevelItem(item)

        # Load files
        self.files_table.setRowCount(0)
        self._gdrive_files = self._service.list_resume_files(self._current_folder_id)

        for file in self._gdrive_files:
            row = self.files_table.rowCount()
            self.files_table.insertRow(row)

            name_item = QTableWidgetItem(file.name)
            name_item.setData(Qt.ItemDataRole.UserRole, file)
            self.files_table.setItem(row, 0, name_item)

            type_item = QTableWidgetItem(file.mime_type.split("/")[-1].upper())
            self.files_table.setItem(row, 1, type_item)

            size = f"{file.size / 1024:.1f} KB" if file.size else "N/A"
            self.files_table.setItem(row, 2, QTableWidgetItem(size))

        self.import_gdrive_btn.setEnabled(len(self._gdrive_files) > 0)

    def _on_folder_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Navigate into a folder."""
        folder_id = item.data(0, Qt.ItemDataRole.UserRole)
        if folder_id:
            self._folder_history.append(self._current_folder_id)
            self._current_folder_id = folder_id
            self.back_btn.setEnabled(True)
            self.path_label.setText(f"/ {item.text(0).replace('📁 ', '')}")
            self._refresh_folder()

    def _go_back(self):
        """Go back to previous folder."""
        if self._folder_history:
            self._current_folder_id = self._folder_history.pop()
            self.back_btn.setEnabled(len(self._folder_history) > 0)
            self.path_label.setText("/ My Drive" if self._current_folder_id == "root" else "/ ...")
            self._refresh_folder()

    def _find_form_folder(self):
        """Find Google Form responses folder."""
        form_name = self.form_name_input.text().strip()
        if not form_name:
            QMessageBox.warning(self, "Input Required", "Please enter the Google Form name.")
            return

        folder_id = self._service.get_forms_response_folder(form_name)
        if folder_id:
            self._folder_history.append(self._current_folder_id)
            self._current_folder_id = folder_id
            self.back_btn.setEnabled(True)
            self.path_label.setText(f"/ {form_name} (File responses)")
            self._refresh_folder()
            QMessageBox.information(
                self,
                "Folder Found",
                f"Found form responses folder!\n\n"
                f"Files: {len(self._gdrive_files)} resume(s) found"
            )
        else:
            QMessageBox.warning(
                self,
                "Not Found",
                f"Could not find folder for form: '{form_name}'\n\n"
                "Make sure:\n"
                "• The form name is exact\n"
                "• You have access to the folder\n"
                "• The form has file upload responses"
            )

    def _select_all_files(self):
        """Select all files in the table."""
        self.files_table.selectAll()

    def _import_selected(self):
        """Download and import selected files."""
        selected_rows = set(item.row() for item in self.files_table.selectedItems())

        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select files to import.")
            return

        # Download files to temp directory
        from src.utils.config import DATA_DIR
        import_dir = DATA_DIR / "imports" / "gdrive"
        import_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []

        # Show progress
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog("Downloading files...", "Cancel", 0, len(selected_rows), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        for i, row in enumerate(selected_rows):
            if progress.wasCanceled():
                break

            file = self._gdrive_files[row]
            progress.setLabelText(f"Downloading: {file.name}")
            progress.setValue(i)

            output_path = import_dir / file.name
            if self._service.download_file(file.id, output_path):
                downloaded_files.append(str(output_path))

        progress.setValue(len(selected_rows))

        if downloaded_files:
            self.import_requested.emit(downloaded_files)


class GoogleSheetsTab(QWidget):
    """Tab for importing candidate metadata from Google Sheets."""

    metadata_imported = pyqtSignal(list)  # List of metadata dicts

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Info box
        info_frame = QFrame()
        info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                padding: 16px;
            }}
        """)
        info_layout = QVBoxLayout(info_frame)

        info_title = QLabel("📊 Google Sheets Integration")
        info_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        info_title.setStyleSheet("color: #1e40af;")
        info_layout.addWidget(info_title)

        info_text = QLabel(
            "Import candidate metadata from Google Sheets linked to Google Forms.\n"
            "This helps match uploaded resumes with student details like:\n"
            "• Name, Roll Number, Branch/Department\n"
            "• Email, Phone, Year of Graduation\n"
            "• CGPA, Preferred Job Roles"
        )
        info_text.setStyleSheet("color: #1e40af;")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        layout.addWidget(info_frame)

        # Sheet URL input
        url_group = QGroupBox("Google Sheet URL")
        url_layout = QVBoxLayout(url_group)

        self.sheet_url_input = QLineEdit()
        self.sheet_url_input.setPlaceholderText("Paste Google Sheets URL here...")
        self.sheet_url_input.setMinimumHeight(40)
        url_layout.addWidget(self.sheet_url_input)

        url_hint = QLabel("Example: https://docs.google.com/spreadsheets/d/SHEET_ID/edit")
        url_hint.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        url_layout.addWidget(url_hint)

        layout.addWidget(url_group)

        # Column mapping
        mapping_group = QGroupBox("Column Mapping")
        mapping_layout = QGridLayout(mapping_group)

        fields = [
            ("Name Column:", "name_col", "A"),
            ("Email Column:", "email_col", "B"),
            ("Roll Number:", "roll_col", "C"),
            ("Branch/Dept:", "branch_col", "D"),
            ("Phone Column:", "phone_col", "E"),
            ("CGPA Column:", "cgpa_col", "F"),
        ]

        self.column_inputs = {}
        for i, (label, key, default) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2

            mapping_layout.addWidget(QLabel(label), row, col)
            input_field = QLineEdit(default)
            input_field.setMaximumWidth(60)
            self.column_inputs[key] = input_field
            mapping_layout.addWidget(input_field, row, col + 1)

        layout.addWidget(mapping_group)

        # Preview area
        preview_label = QLabel("Data Preview:")
        preview_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(preview_label)

        self.preview_table = QTableWidget()
        self.preview_table.setMinimumHeight(200)
        self.preview_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['surface']};
                border: 1px solid #e2e8f0;
                border-radius: 6px;
            }}
        """)
        layout.addWidget(self.preview_table)

        # Buttons
        btn_layout = QHBoxLayout()

        fetch_btn = QPushButton("🔄 Fetch Data")
        fetch_btn.setMinimumHeight(40)
        fetch_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #3367d6;
            }}
        """)
        fetch_btn.clicked.connect(self._fetch_sheet_data)
        btn_layout.addWidget(fetch_btn)

        btn_layout.addStretch()

        self.import_meta_btn = QPushButton("Import Metadata")
        self.import_meta_btn.setMinimumHeight(40)
        self.import_meta_btn.setEnabled(False)
        self.import_meta_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1d4ed8;
            }}
            QPushButton:disabled {{
                background-color: #cbd5e1;
            }}
        """)
        self.import_meta_btn.clicked.connect(self._import_metadata)
        btn_layout.addWidget(self.import_meta_btn)

        layout.addLayout(btn_layout)

        self._sheet_data = []

    def _fetch_sheet_data(self):
        """Fetch data from Google Sheets."""
        url = self.sheet_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Required", "Please enter the Google Sheets URL.")
            return

        # Extract sheet ID from URL
        import re
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
        if not match:
            QMessageBox.warning(
                self,
                "Invalid URL",
                "Could not extract Sheet ID from URL.\n"
                "Make sure you're using the full Google Sheets URL."
            )
            return

        sheet_id = match.group(1)

        try:
            from src.services.google_drive_service import get_drive_service
            service = get_drive_service()

            if not service.is_authenticated():
                if not service.authenticate():
                    QMessageBox.warning(self, "Auth Required", "Please connect to Google Drive first.")
                    return

            # Note: This requires Google Sheets API, showing placeholder
            QMessageBox.information(
                self,
                "Feature Note",
                f"Sheet ID: {sheet_id}\n\n"
                "To fully implement Google Sheets reading, you need to:\n"
                "1. Enable Google Sheets API in Cloud Console\n"
                "2. Install: pip install gspread\n\n"
                "For now, you can export the sheet as CSV and import locally."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {e}")

    def _import_metadata(self):
        """Import metadata to update candidates."""
        if self._sheet_data:
            self.metadata_imported.emit(self._sheet_data)


class ImportCenterDialog(QDialog):
    """Main Import Center dialog with multiple import methods."""

    candidates_imported = pyqtSignal(list)  # List of candidate dicts

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Center")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(f"background-color: {COLORS['background']};")

        self._imported_candidates = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-bottom: 1px solid #e2e8f0;
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 16)

        title = QLabel("📥 Import Center")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        header_layout.addWidget(title)

        header_layout.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(32, 32)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-size: 18px;
                color: #64748b;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
                border-radius: 4px;
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)

        layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {COLORS['background']};
            }}
            QTabBar::tab {{
                padding: 12px 24px;
                margin-right: 4px;
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {COLORS['primary']};
                border-bottom: 2px solid {COLORS['primary']};
                font-weight: bold;
            }}
            QTabBar::tab:hover {{
                background-color: #f1f5f9;
            }}
        """)

        # Local import tab
        self.local_tab = LocalImportTab()
        self.local_tab.import_requested.connect(self._process_files)
        self.tabs.addTab(self.local_tab, "📁 Local Files")

        # Google Drive tab
        self.gdrive_tab = GoogleDriveTab()
        self.gdrive_tab.import_requested.connect(self._process_files)
        self.tabs.addTab(self.gdrive_tab, "☁️ Google Drive")

        # Google Sheets tab
        self.gsheets_tab = GoogleSheetsTab()
        self.gsheets_tab.metadata_imported.connect(self._handle_metadata)
        self.tabs.addTab(self.gsheets_tab, "📊 Google Sheets")

        layout.addWidget(self.tabs)

        # Progress bar (hidden by default)
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        self.progress_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-top: 1px solid #e2e8f0;
            }}
        """)
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(24, 16, 24, 16)

        self.progress_label = QLabel("Processing...")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                background-color: #e2e8f0;
                border-radius: 4px;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 4px;
            }}
        """)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(self.progress_frame)

        # Footer with results
        self.footer = QFrame()
        self.footer.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-top: 1px solid #e2e8f0;
            }}
        """)
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(24, 16, 24, 16)

        self.result_label = QLabel("")
        self.result_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        footer_layout.addWidget(self.result_label)

        footer_layout.addStretch()

        self.done_btn = QPushButton("Done")
        self.done_btn.setMinimumHeight(40)
        self.done_btn.setMinimumWidth(120)
        self.done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.done_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1d4ed8;
            }}
        """)
        self.done_btn.clicked.connect(self._finish_import)
        footer_layout.addWidget(self.done_btn)

        layout.addWidget(self.footer)

    def _process_files(self, file_paths: list):
        """Process files using the import worker."""
        self.progress_frame.setVisible(True)
        self.progress_bar.setMaximum(len(file_paths))
        self.progress_bar.setValue(0)

        self._worker = ImportWorker(file_paths)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_processed.connect(self._on_file_processed)
        self._worker.finished.connect(self._on_import_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"Processing ({current}/{total}): {message}")

    def _on_file_processed(self, result: dict):
        if result.get("success"):
            self._imported_candidates.append(result)

    def _on_import_finished(self, success_count: int, error_count: int):
        self.progress_frame.setVisible(False)
        self.result_label.setText(
            f"✓ Imported: {success_count} | ✗ Errors: {error_count}"
        )
        self.result_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: 500;")

        if success_count > 0:
            QMessageBox.information(
                self,
                "Import Complete",
                f"Successfully imported {success_count} candidate(s).\n"
                f"Errors: {error_count}\n\n"
                "Click 'Done' to add them to the database."
            )

    def _handle_metadata(self, metadata: list):
        """Handle metadata imported from Google Sheets."""
        # This would merge metadata with existing candidates
        pass

    def _finish_import(self):
        """Finish import and emit results."""
        if self._imported_candidates:
            self.candidates_imported.emit(self._imported_candidates)
        self.accept()

    def get_imported_candidates(self) -> list:
        """Get the list of imported candidates."""
        return self._imported_candidates
