from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QTextEdit,
    QVBoxLayout,
)


MIN_REASON_LENGTH: int = 10
SCORE_STEP: float = 0.01


class OverrideScoreDialog(QDialog):
    """
    Modal for capturing a manual score override + reason.

    Returns via .accepted() — callers read .new_score and .reason after
    exec() returns QDialog.DialogCode.Accepted.
    """

    def __init__(
        self,
        *,
        candidate_name: str,
        job_title: str,
        current_score: float,
        ai_score: float,
        existing_reason: Optional[str] = None,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.setWindowTitle("Override Match Score")
        self.setModal(True)
        self.setMinimumWidth(460)

        self._ai_score: float = max(0.0, min(1.0, float(ai_score)))
        initial_score: float = max(0.0, min(1.0, float(current_score)))

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
        header: QLabel = QLabel(f"<b>{candidate_name}</b> — {job_title}")
        header.setWordWrap(True)
        layout.addWidget(header)

        ai_line: QLabel = QLabel(
            f"AI score: <b>{self._ai_score:.0%}</b> · "
            f"Current effective score: <b>{initial_score:.0%}</b>"
        )
        ai_line.setStyleSheet("color: #5c5c5c;")
        layout.addWidget(ai_line)

        # Score controls — slider + spinbox bound both ways
        score_row: QFormLayout = QFormLayout()
        score_row.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        score_row.setSpacing(10)

        self._slider: QSlider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setValue(int(round(initial_score * 100)))
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(10)

        self._spin: QDoubleSpinBox = QDoubleSpinBox()
        self._spin.setRange(0.0, 1.0)
        self._spin.setDecimals(2)
        self._spin.setSingleStep(SCORE_STEP)
        self._spin.setValue(initial_score)

        slider_wrap = QHBoxLayout()
        slider_wrap.addWidget(self._slider, 1)
        slider_wrap.addWidget(self._spin, 0)
        score_row.addRow("New score:", slider_wrap)

        layout.addLayout(score_row)

        # Reason
        reason_label: QLabel = QLabel(
            f"Reason (min {MIN_REASON_LENGTH} characters) — this is logged in the audit trail:"
        )
        layout.addWidget(reason_label)

        self._reason_edit: QTextEdit = QTextEdit()
        self._reason_edit.setFont(QFont("Monospace", 11))
        self._reason_edit.setMinimumHeight(110)
        if existing_reason:
            self._reason_edit.setPlainText(existing_reason)
        layout.addWidget(self._reason_edit)

        # Live validation summary
        self._validation_label: QLabel = QLabel("")
        self._validation_label.setStyleSheet("color: #c0392b; font-size: 11px;")
        layout.addWidget(self._validation_label)

        # Buttons
        self._buttons: QDialogButtonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save_btn = self._buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_btn is not None:
            save_btn.setText("Save override")
        layout.addWidget(self._buttons)

        # Signal wiring
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)
        self._reason_edit.textChanged.connect(self._revalidate)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)

        self._revalidate()

    # Public result accessors — read after exec() returns Accepted
    @property
    def new_score(self) -> float:
        return float(self._spin.value())

    @property
    def reason(self) -> str:
        return self._reason_edit.toPlainText().strip()

    # Internal — signal handlers
    def _on_slider_changed(self, value: int) -> None:
        new: float = value / 100.0
        if abs(new - self._spin.value()) > 1e-6:
            self._spin.blockSignals(True)
            self._spin.setValue(new)
            self._spin.blockSignals(False)
        self._revalidate()

    def _on_spin_changed(self, value: float) -> None:
        slider_val: int = int(round(value * 100))
        if slider_val != self._slider.value():
            self._slider.blockSignals(True)
            self._slider.setValue(slider_val)
            self._slider.blockSignals(False)
        self._revalidate()

    def _revalidate(self) -> None:
        reason: str = self._reason_edit.toPlainText().strip()
        errors: list[str] = []
        if len(reason) < MIN_REASON_LENGTH:
            errors.append(
                f"Reason must be at least {MIN_REASON_LENGTH} characters "
                f"({len(reason)}/{MIN_REASON_LENGTH})"
            )
        score: float = self._spin.value()
        if not (0.0 <= score <= 1.0):
            errors.append("Score must be between 0.00 and 1.00")
        self._validation_label.setText(" · ".join(errors))

        save_btn = self._buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_btn is not None:
            save_btn.setEnabled(len(errors) == 0)

    def _on_accept(self) -> None:
        # Defensive: even if the button is accidentally enabled, block accept
        self._revalidate()
        if self._validation_label.text():
            return
        self.accept()
