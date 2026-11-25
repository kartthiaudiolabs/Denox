# Copyright 2026 Inworld AI & Antigravity
"""Main entry point for the Inworld Neo-Isolator Mac Application."""

import argparse
import logging
import socket
import sys
import threading
import time
import uvicorn
import webview

from mac_app import server

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mac_app")


def find_free_port(start_port: int = 7888) -> int:
    """Finds an available local port starting from start_port."""
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


class NativeAppBridge:
    """Exposes native macOS Cocoa dialogs to the JavaScript frontend."""

    def __init__(self, window):
        self.window = window

    def select_file(self) -> str:
        """Opens native macOS open-file dialog."""
        if not self.window:
            return ""
        file_types = ("Audio Files (*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.aac;*.aiff)", "All Files (*.*)")
        result = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=file_types
        )
        if result and len(result) > 0:
            return result[0]
        return ""

    def select_folder(self) -> str:
        """Opens native macOS folder dialog."""
        if not self.window:
            return ""
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if result and len(result) > 0:
            return result[0]
        return ""


def run_app():
    parser = argparse.ArgumentParser(description="Denox // Quantum Audio Studio Mac Application")
    parser.add_argument("--port", type=int, default=None, help="Server port (defaults to next free port)")
    parser.add_argument("--server-only", action="store_true", help="Run server only without native Cocoa window")
    args = parser.parse_args()

    port = args.port if args.port else find_free_port(7888)
    app_url = f"http://127.0.0.1:{port}"

    # Start FastAPI server in a background daemon thread
    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning")
    uv_server = uvicorn.Server(config)

    server_thread = threading.Thread(target=uv_server.run, daemon=True)
    server_thread.start()

    logger.info("Server running at %s", app_url)
    # Wait briefly for server startup
    time.sleep(0.5)

    if args.server_only:
        logger.info("Running in server-only mode. Open %s in your browser.", app_url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting...")
            sys.exit(0)

    # Launch native macOS Cocoa window with PyWebView
    logger.info("Launching native macOS Cocoa window...")
    bridge_container = [None]

    window = webview.create_window(
        title="Denox // Quantum Audio Studio",
        url=app_url,
        width=1240,
        height=840,
        min_size=(1000, 680),
        background_color="#06080D",
        text_select=False
    )
    bridge = NativeAppBridge(window)
    window.expose(bridge.select_file, bridge.select_folder)

    webview.start(debug=False)
    logger.info("Native window closed. Exiting application.")


if __name__ == "__main__":
    run_app()
