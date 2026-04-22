from __future__ import annotations

import getpass
from datetime import datetime, timezone
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.utils.constants import COLORS
from src.ui.widgets.buttons import DangerButton, PrimaryButton, SecondaryButton


# ── Workspace Selector ─────────────────────────────────────────────────────────

class WorkspaceSelectorDialog(QDialog):
    """List recent workspaces — switch, create new, or archive."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Switch Workspace")
        self.setMinimumSize(540, 440)
        self._selected_workspace: Optional[object] = None
        self._workspaces: list[object] = []
        self._build_ui()
        self._load()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        subtitle = QLabel("Select a workspace or create a new one")
        subtitle.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        layout.addWidget(subtitle)

        self._list = QListWidget()
        self._list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 8px;
                padding: 4px;
                outline: none;
            }}
            QListWidget::item {{
                color: {COLORS['text_primary']};
                padding: 10px 12px;
                border-radius: 6px;
                margin: 1px 2px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary_glow']};
                color: {COLORS['primary']};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {COLORS['surface_overlay']};
            }}
            """
        )
        self._list.itemDoubleClicked.connect(self._on_switch)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()

        self._new_btn = SecondaryButton("+ New Workspace")
        self._new_btn.clicked.connect(self._on_new)
        btn_row.addWidget(self._new_btn)

        self._archive_btn = SecondaryButton("Archive Selected")
        self._archive_btn.setEnabled(False)
        self._archive_btn.clicked.connect(self._on_archive)
        btn_row.addWidget(self._archive_btn)

        btn_row.addStretch()

        self._switch_btn = PrimaryButton("Switch")
        self._switch_btn.setEnabled(False)
        self._switch_btn.clicked.connect(self._on_switch)
        btn_row.addWidget(self._switch_btn)

        layout.addLayout(btn_row)

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            from src.data.sql.repositories import (
                get_job_record_repository,
                get_workspace_repository,
            )
            ws_repo = get_workspace_repository()
            job_repo = get_job_record_repository()
            self._workspaces = ws_repo.list_recent(limit=20)
        except Exception:
            self._workspaces = []
            job_repo = None

        self._list.clear()

        if not self._workspaces:
            placeholder = QListWidgetItem(
                "No workspaces yet — click '+ New Workspace' to get started"
            )
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            placeholder.setForeground(
                __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(
                    COLORS["text_tertiary"]
                )
            )
            self._list.addItem(placeholder)
            return

        for ws in self._workspaces:
            job_count = 0
            if job_repo is not None:
                try:
                    job_count = job_repo.count_for_workspace(ws.id)  # type: ignore[attr-defined]
                except Exception:
                    pass

            last_opened = ws.last_opened_at  # type: ignore[attr-defined]
            age_str = ""
            if last_opened is not None:
                delta = datetime.now(timezone.utc) - last_opened
                age_str = " · today" if delta.days == 0 else f" · {delta.days}d ago"

            jobs_str = f"{job_count} job" + ("s" if job_count != 1 else "")
            label = (
                f"{ws.name}   "  # type: ignore[attr-defined]
                f"[{ws.status.value}]{age_str}   ({jobs_str})"  # type: ignore[attr-defined]
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, ws)
            self._list.addItem(item)

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        items = self._list.selectedItems()
        if not items:
            self._switch_btn.setEnabled(False)
            self._archive_btn.setEnabled(False)
            return
        ws = items[0].data(Qt.ItemDataRole.UserRole)
        if ws is None:
            self._switch_btn.setEnabled(False)
            self._archive_btn.setEnabled(False)
            return
        from src.data.sql.models.workspace import WorkspaceStatus
        self._switch_btn.setEnabled(True)
        self._archive_btn.setEnabled(ws.status == WorkspaceStatus.ACTIVE)  # type: ignore[attr-defined]

    def _on_switch(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        ws = items[0].data(Qt.ItemDataRole.UserRole)
        if ws is None:
            return
        try:
            from src.data.sql.repositories import get_workspace_repository
            get_workspace_repository().touch(ws.id)  # type: ignore[attr-defined]
        except Exception:
            pass
        self._selected_workspace = ws
        self.accept()

    def _on_new(self) -> None:
        dlg = CreateWorkspaceDialog(parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            from src.data.sql.repositories import get_workspace_repository
            ws = get_workspace_repository().create_workspace(
                name=dlg.workspace_name,
                description=dlg.workspace_description or None,
                created_by=_current_user(),
            )
            self._selected_workspace = ws
            self.accept()
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Could not create workspace:\n{exc}"
            )

    def _on_archive(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        ws = items[0].data(Qt.ItemDataRole.UserRole)
        if ws is None:
            return
        reply = QMessageBox.question(
            self,
            "Archive Workspace",
            f'Archive "{ws.name}"?\n\n'  # type: ignore[attr-defined]
            "Archived workspaces are read-only. You can restore them later from Settings.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from src.data.sql.repositories import get_workspace_repository
            get_workspace_repository().archive(ws.id)  # type: ignore[attr-defined]
            self._load()
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Could not archive workspace:\n{exc}"
            )

    # ── Public result ──────────────────────────────────────────────────────────

    @property
    def selected_workspace(self) -> Optional[object]:
        return self._selected_workspace


# ── Create Workspace ───────────────────────────────────────────────────────────

class CreateWorkspaceDialog(QDialog):
    """Name + optional description for a new workspace."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Workspace")
        self.setFixedWidth(440)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Senior Engineers — Q3 2026")
        self._name_edit.setMinimumHeight(34)
        self._name_edit.textChanged.connect(self._validate)
        form.addRow("Name *", self._name_edit)

        self._desc_edit = QPlainTextEdit()
        self._desc_edit.setPlaceholderText("Optional description…")
        self._desc_edit.setFixedHeight(80)
        form.addRow("Description", self._desc_edit)

        layout.addLayout(form)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet(
            f"color: {COLORS['error']}; font-size: 11px;"
        )
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
        self._save_btn.setEnabled(False)  # type: ignore[union-attr]
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            self._error_label.setText("Name is required")
            self._error_label.setVisible(True)
            self._save_btn.setEnabled(False)  # type: ignore[union-attr]
        elif len(name) > 200:
            self._error_label.setText("Name must be 200 characters or fewer")
            self._error_label.setVisible(True)
            self._save_btn.setEnabled(False)  # type: ignore[union-attr]
        else:
            self._error_label.setVisible(False)
            self._save_btn.setEnabled(True)  # type: ignore[union-attr]

    @property
    def workspace_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def workspace_description(self) -> str:
        return self._desc_edit.toPlainText().strip()


# ── Purge Confirmation ─────────────────────────────────────────────────────────

class PurgeWorkspaceDialog(QDialog):
    """User must type the exact workspace name to confirm permanent deletion."""

    def __init__(
        self,
        workspace_name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Purge Workspace")
        self.setFixedWidth(480)
        self._workspace_name = workspace_name
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        warning = QLabel(
            f"⚠  This will permanently delete all data for\n"
            f'"{self._workspace_name}":\n\n'
            "  • All job records\n"
            "  • All match scores (including manual overrides)\n\n"
            "Audit logs are preserved (workspace reference set to null).\n"
            "This cannot be undone."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 12px; padding: 14px;"
            f" background-color: {COLORS['warning_dim']}; border-radius: 8px;"
        )
        layout.addWidget(warning)

        confirm_label = QLabel(
            f"Type the workspace name to confirm:"
        )
        confirm_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        layout.addWidget(confirm_label)

        self._confirm_edit = QLineEdit()
        self._confirm_edit.setPlaceholderText(self._workspace_name)
        self._confirm_edit.setMinimumHeight(34)
        self._confirm_edit.textChanged.connect(self._validate)
        layout.addWidget(self._confirm_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = SecondaryButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._purge_btn = DangerButton("Purge Permanently")
        self._purge_btn.setEnabled(False)
        self._purge_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._purge_btn)

        layout.addLayout(btn_row)

    def _validate(self) -> None:
        self._purge_btn.setEnabled(
            self._confirm_edit.text().strip() == self._workspace_name
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
