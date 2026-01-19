"""
AI-ATS Main Entry Point

This module serves as the main entry point for the AI-ATS application.
It initializes all components and launches the GUI.
"""

import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    """
    Main entry point for the AI-ATS application.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        # Initialize logging first
        from src.utils.logger import setup_logging, log

        setup_logging()
        log.info("Starting AI-ATS application...")

        # Load configuration
        from src.utils.config import get_settings

        settings = get_settings()
        log.info(f"Environment: {settings.environment}")
        log.info(f"Debug mode: {settings.debug}")

        # Initialize database connection
        log.info("Initializing database connection...")
        # TODO: Initialize MongoDB connection

        # Initialize ML models
        log.info("Loading ML models...")
        # TODO: Load embedding models and NLP pipeline

        # Launch GUI
        log.info("Launching user interface...")
        from src.ui.main_window import run_application

        return run_application()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
