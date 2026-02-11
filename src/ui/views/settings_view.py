"""
Settings view for AI-ATS application.

Provides interface for configuring application settings.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QTabWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS, DEFAULT_SCORING_WEIGHTS
from src.ui.views.base_view import BaseView
from src.ui.widgets import (
    Card,
    InfoCard,
    PrimaryButton,
    SecondaryButton,
)


class SettingsView(BaseView):
    """
    Application settings view.

    Provides configuration for:
    - General settings (theme, language)
    - Matching settings (weights, thresholds)
    - Database settings
    - AI/ML settings
    """

    def __init__(self, parent=None):
        """Initialize the settings view."""
        super().__init__(
            title="Settings",
            description="Configure application settings and preferences",
            parent=parent,
        )
        self._setup_settings_view()
        self._load_current_settings()

    def _setup_settings_view(self):
        """Set up the settings view content."""
        # Create tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: {COLORS['surface']};
                padding: 16px;
            }}
            QTabBar::tab {{
                background-color: #f1f5f9;
                color: {COLORS['text_secondary']};
                padding: 10px 20px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['surface']};
                color: {COLORS['primary']};
                font-weight: 500;
            }}
            QTabBar::tab:hover {{
                background-color: #e2e8f0;
            }}
        """)

        # General settings tab
        general_tab = self._create_general_tab()
        tabs.addTab(general_tab, "General")

        # Matching settings tab
        matching_tab = self._create_matching_tab()
        tabs.addTab(matching_tab, "Matching")

        # AI/ML settings tab
        ml_tab = self._create_ml_tab()
        tabs.addTab(ml_tab, "AI / ML")

        # Database settings tab
        db_tab = self._create_database_tab()
        tabs.addTab(db_tab, "Database")

        self.add_widget(tabs)

        # Save button
        save_layout = QHBoxLayout()
        save_layout.addStretch()

        reset_btn = SecondaryButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        save_layout.addWidget(reset_btn)

        save_btn = PrimaryButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        save_layout.addWidget(save_btn)

        self.add_layout(save_layout)

    def _create_general_tab(self) -> QWidget:
        """Create general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_group.setStyleSheet(self._group_style())
        appearance_layout = QFormLayout(appearance_group)
        appearance_layout.setSpacing(12)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "System"])
        self.theme_combo.setMinimumHeight(32)
        appearance_layout.addRow("Theme:", self.theme_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 18)
        self.font_size_spin.setValue(12)
        self.font_size_spin.setSuffix(" pt")
        appearance_layout.addRow("Font Size:", self.font_size_spin)

        layout.addWidget(appearance_group)

        # Language group
        lang_group = QGroupBox("Language & Region")
        lang_group.setStyleSheet(self._group_style())
        lang_layout = QFormLayout(lang_group)
        lang_layout.setSpacing(12)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Spanish", "French", "German"])
        self.lang_combo.setMinimumHeight(32)
        lang_layout.addRow("Language:", self.lang_combo)

        layout.addWidget(lang_group)

        # Window group
        window_group = QGroupBox("Window")
        window_group.setStyleSheet(self._group_style())
        window_layout = QFormLayout(window_group)
        window_layout.setSpacing(12)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(800, 2560)
        self.width_spin.setValue(1400)
        self.width_spin.setSuffix(" px")
        window_layout.addRow("Default Width:", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(600, 1440)
        self.height_spin.setValue(900)
        self.height_spin.setSuffix(" px")
        window_layout.addRow("Default Height:", self.height_spin)

        layout.addWidget(window_group)

        layout.addStretch()
        return tab

    def _create_matching_tab(self) -> QWidget:
        """Create matching settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Scoring weights group
        weights_group = QGroupBox("Scoring Weights")
        weights_group.setStyleSheet(self._group_style())
        weights_layout = QFormLayout(weights_group)
        weights_layout.setSpacing(12)

        info_label = QLabel("Adjust how different factors contribute to the match score. Total should equal 100%.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; margin-bottom: 8px;")
        weights_layout.addRow(info_label)

        self.skills_weight = QSpinBox()
        self.skills_weight.setRange(0, 100)
        self.skills_weight.setValue(35)
        self.skills_weight.setSuffix(" %")
        weights_layout.addRow("Skills Match:", self.skills_weight)

        self.experience_weight = QSpinBox()
        self.experience_weight.setRange(0, 100)
        self.experience_weight.setValue(25)
        self.experience_weight.setSuffix(" %")
        weights_layout.addRow("Experience Match:", self.experience_weight)

        self.education_weight = QSpinBox()
        self.education_weight.setRange(0, 100)
        self.education_weight.setValue(15)
        self.education_weight.setSuffix(" %")
        weights_layout.addRow("Education Match:", self.education_weight)

        self.semantic_weight = QSpinBox()
        self.semantic_weight.setRange(0, 100)
        self.semantic_weight.setValue(20)
        self.semantic_weight.setSuffix(" %")
        weights_layout.addRow("Semantic Similarity:", self.semantic_weight)

        self.keyword_weight = QSpinBox()
        self.keyword_weight.setRange(0, 100)
        self.keyword_weight.setValue(5)
        self.keyword_weight.setSuffix(" %")
        weights_layout.addRow("Keyword Match:", self.keyword_weight)

        layout.addWidget(weights_group)

        # Thresholds group
        threshold_group = QGroupBox("Score Thresholds")
        threshold_group.setStyleSheet(self._group_style())
        threshold_layout = QFormLayout(threshold_group)
        threshold_layout.setSpacing(12)

        self.excellent_threshold = QDoubleSpinBox()
        self.excellent_threshold.setRange(0, 1)
        self.excellent_threshold.setSingleStep(0.05)
        self.excellent_threshold.setValue(0.85)
        threshold_layout.addRow("Excellent (≥):", self.excellent_threshold)

        self.good_threshold = QDoubleSpinBox()
        self.good_threshold.setRange(0, 1)
        self.good_threshold.setSingleStep(0.05)
        self.good_threshold.setValue(0.70)
        threshold_layout.addRow("Good (≥):", self.good_threshold)

        self.fair_threshold = QDoubleSpinBox()
        self.fair_threshold.setRange(0, 1)
        self.fair_threshold.setSingleStep(0.05)
        self.fair_threshold.setValue(0.50)
        threshold_layout.addRow("Fair (≥):", self.fair_threshold)

        layout.addWidget(threshold_group)

        layout.addStretch()
        return tab

    def _create_ml_tab(self) -> QWidget:
        """Create AI/ML settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Embedding model group
        embedding_group = QGroupBox("Embedding Model")
        embedding_group.setStyleSheet(self._group_style())
        embedding_layout = QFormLayout(embedding_group)
        embedding_layout.setSpacing(12)

        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.addItems([
            "all-MiniLM-L6-v2 (Fast)",
            "all-mpnet-base-v2 (Balanced)",
            "all-MiniLM-L12-v2 (Quality)",
        ])
        embedding_layout.addRow("Model:", self.embedding_model_combo)

        self.embedding_device_combo = QComboBox()
        self.embedding_device_combo.addItems(["Auto", "CPU", "CUDA", "MPS"])
        embedding_layout.addRow("Device:", self.embedding_device_combo)

        layout.addWidget(embedding_group)

        # Bias detection group
        bias_group = QGroupBox("Bias Detection")
        bias_group.setStyleSheet(self._group_style())
        bias_layout = QFormLayout(bias_group)
        bias_layout.setSpacing(12)

        self.bias_detection_check = QCheckBox("Enable bias detection")
        self.bias_detection_check.setChecked(True)
        bias_layout.addRow(self.bias_detection_check)

        self.auto_mitigate_check = QCheckBox("Auto-apply bias mitigation")
        self.auto_mitigate_check.setChecked(False)
        bias_layout.addRow(self.auto_mitigate_check)

        layout.addWidget(bias_group)

        # Explainability group
        explain_group = QGroupBox("Explainability")
        explain_group.setStyleSheet(self._group_style())
        explain_layout = QFormLayout(explain_group)
        explain_layout.setSpacing(12)

        self.lime_check = QCheckBox("Generate LIME explanations")
        self.lime_check.setChecked(True)
        explain_layout.addRow(self.lime_check)

        self.shap_check = QCheckBox("Generate SHAP values")
        self.shap_check.setChecked(True)
        explain_layout.addRow(self.shap_check)

        layout.addWidget(explain_group)

        layout.addStretch()
        return tab

    def _create_database_tab(self) -> QWidget:
        """Create database settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # MongoDB settings group
        mongo_group = QGroupBox("MongoDB Connection")
        mongo_group.setStyleSheet(self._group_style())
        mongo_layout = QFormLayout(mongo_group)
        mongo_layout.setSpacing(12)

        self.db_host_input = QLineEdit()
        self.db_host_input.setText("localhost")
        self.db_host_input.setMinimumHeight(32)
        mongo_layout.addRow("Host:", self.db_host_input)

        self.db_port_spin = QSpinBox()
        self.db_port_spin.setRange(1, 65535)
        self.db_port_spin.setValue(27017)
        mongo_layout.addRow("Port:", self.db_port_spin)

        self.db_name_input = QLineEdit()
        self.db_name_input.setText("ai_ats")
        self.db_name_input.setMinimumHeight(32)
        mongo_layout.addRow("Database:", self.db_name_input)

        self.db_user_input = QLineEdit()
        self.db_user_input.setPlaceholderText("Optional")
        self.db_user_input.setMinimumHeight(32)
        mongo_layout.addRow("Username:", self.db_user_input)

        self.db_pass_input = QLineEdit()
        self.db_pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.db_pass_input.setPlaceholderText("Optional")
        self.db_pass_input.setMinimumHeight(32)
        mongo_layout.addRow("Password:", self.db_pass_input)

        test_btn = SecondaryButton("Test Connection")
        test_btn.clicked.connect(self._test_db_connection)
        mongo_layout.addRow("", test_btn)

        layout.addWidget(mongo_group)

        # Vector store settings
        vector_group = QGroupBox("Vector Store")
        vector_group.setStyleSheet(self._group_style())
        vector_layout = QFormLayout(vector_group)
        vector_layout.setSpacing(12)

        self.vector_provider_combo = QComboBox()
        self.vector_provider_combo.addItems(["ChromaDB", "FAISS"])
        vector_layout.addRow("Provider:", self.vector_provider_combo)

        layout.addWidget(vector_group)

        layout.addStretch()
        return tab

    def _group_style(self) -> str:
        """Return common group box style."""
        return f"""
            QGroupBox {{
                font-weight: 600;
                font-size: 13px;
                color: {COLORS['text_primary']};
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 12px;
                padding: 16px;
                padding-top: 24px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {COLORS['surface']};
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 6px 10px;
                min-height: 28px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
        """

    def _load_current_settings(self):
        """Load current settings from AppSettings into the UI."""
        try:
            from src.utils.config import get_settings
            settings = get_settings()

            # General tab
            theme_map = {"light": 0, "dark": 1, "system": 2}
            self.theme_combo.setCurrentIndex(theme_map.get(settings.ui.theme, 2))
            self.font_size_spin.setValue(settings.ui.font_size)
            self.width_spin.setValue(settings.ui.window_width)
            self.height_spin.setValue(settings.ui.window_height)

            # Matching tab - weights from DEFAULT_SCORING_WEIGHTS
            self.skills_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("skills_match", 0.35) * 100))
            self.experience_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("experience_match", 0.25) * 100))
            self.education_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("education_match", 0.15) * 100))
            self.semantic_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("semantic_similarity", 0.20) * 100))
            self.keyword_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("keyword_match", 0.05) * 100))

            # ML tab
            model_map = {
                "sentence-transformers/all-MiniLM-L6-v2": 0,
                "sentence-transformers/all-mpnet-base-v2": 1,
                "sentence-transformers/all-MiniLM-L12-v2": 2,
            }
            self.embedding_model_combo.setCurrentIndex(
                model_map.get(settings.ml.embedding_model, 0)
            )
            device_map = {"auto": 0, "cpu": 1, "cuda": 2, "mps": 3}
            self.embedding_device_combo.setCurrentIndex(
                device_map.get(settings.ml.device, 0)
            )

            # Database tab
            self.db_host_input.setText(settings.database.host)
            self.db_port_spin.setValue(settings.database.port)
            self.db_name_input.setText(settings.database.name)
            if settings.database.username:
                self.db_user_input.setText(settings.database.username)
            if settings.database.password:
                self.db_pass_input.setText(settings.database.password)

            # Vector store
            provider_map = {"chromadb": 0, "faiss": 1}
            self.vector_provider_combo.setCurrentIndex(
                provider_map.get(settings.vector_store.provider, 0)
            )

        except Exception as e:
            print(f"Error loading settings: {e}")

    def _save_settings(self):
        """Save current settings to environment and reload."""
        # Validate weight totals
        total_weight = (
            self.skills_weight.value() +
            self.experience_weight.value() +
            self.education_weight.value() +
            self.semantic_weight.value() +
            self.keyword_weight.value()
        )

        if total_weight != 100:
            QMessageBox.warning(
                self,
                "Invalid Weights",
                f"Scoring weights must total 100%.\nCurrent total: {total_weight}%",
            )
            return

        import os

        # UI settings
        theme_values = ["light", "dark", "system"]
        os.environ["UI_THEME"] = theme_values[self.theme_combo.currentIndex()]
        os.environ["UI_FONT_SIZE"] = str(self.font_size_spin.value())
        os.environ["UI_WINDOW_WIDTH"] = str(self.width_spin.value())
        os.environ["UI_WINDOW_HEIGHT"] = str(self.height_spin.value())

        # Database settings
        os.environ["DB_HOST"] = self.db_host_input.text().strip()
        os.environ["DB_PORT"] = str(self.db_port_spin.value())
        os.environ["DB_NAME"] = self.db_name_input.text().strip()
        if self.db_user_input.text().strip():
            os.environ["DB_USERNAME"] = self.db_user_input.text().strip()
        if self.db_pass_input.text().strip():
            os.environ["DB_PASSWORD"] = self.db_pass_input.text().strip()

        # ML settings
        model_values = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
        ]
        os.environ["ML_EMBEDDING_MODEL"] = model_values[self.embedding_model_combo.currentIndex()]
        device_values = ["auto", "cpu", "cuda", "mps"]
        os.environ["ML_DEVICE"] = device_values[self.embedding_device_combo.currentIndex()]

        # Vector store
        provider_values = ["chromadb", "faiss"]
        os.environ["VECTOR_PROVIDER"] = provider_values[self.vector_provider_combo.currentIndex()]

        # Reload settings singleton
        try:
            from src.utils.config import reload_settings
            reload_settings()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not reload settings: {e}")
            return

        QMessageBox.information(
            self,
            "Settings Saved",
            "Your settings have been saved and applied.",
        )

    def _reset_defaults(self):
        """Reset settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Reset weights
            self.skills_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("skills_match", 0.35) * 100))
            self.experience_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("experience_match", 0.25) * 100))
            self.education_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("education_match", 0.15) * 100))
            self.semantic_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("semantic_similarity", 0.20) * 100))
            self.keyword_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("keyword_match", 0.05) * 100))

            # Reset thresholds
            self.excellent_threshold.setValue(0.85)
            self.good_threshold.setValue(0.70)
            self.fair_threshold.setValue(0.50)

            # Reset other settings
            self.theme_combo.setCurrentIndex(2)  # System
            self.font_size_spin.setValue(12)
            self.width_spin.setValue(1400)
            self.height_spin.setValue(900)

            # Reset DB
            self.db_host_input.setText("localhost")
            self.db_port_spin.setValue(27017)
            self.db_name_input.setText("ai_ats")
            self.db_user_input.clear()
            self.db_pass_input.clear()

            # Reset ML
            self.embedding_model_combo.setCurrentIndex(0)
            self.embedding_device_combo.setCurrentIndex(0)
            self.vector_provider_combo.setCurrentIndex(0)

    def _test_db_connection(self):
        """Test database connection using current field values."""
        host = self.db_host_input.text().strip()
        port = self.db_port_spin.value()
        db_name = self.db_name_input.text().strip()

        try:
            from urllib.parse import quote_plus
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure

            # Build URI with URL-encoded credentials
            username = self.db_user_input.text().strip()
            password = self.db_pass_input.text().strip()
            if username and password:
                uri = f"mongodb://{quote_plus(username)}:{quote_plus(password)}@{host}:{port}/{db_name}"
            else:
                uri = f"mongodb://{host}:{port}/{db_name}"

            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
            client.close()

            QMessageBox.information(
                self,
                "Connection Successful",
                f"Successfully connected to MongoDB.\n\n"
                f"Host: {host}\n"
                f"Port: {port}\n"
                f"Database: {db_name}",
            )
        except ConnectionFailure:
            QMessageBox.critical(
                self,
                "Connection Failed",
                f"Could not connect to MongoDB at {host}:{port}.\n"
                "Please check the host, port, and credentials.",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Connection Error",
                f"Error testing connection: {e}",
            )
