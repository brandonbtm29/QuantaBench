"""Start-up side of the Sketch dock: is Qt WebEngine there, and its URL scheme registered in time?

Schrödinger's Sketcher runs as WebAssembly in a ``QWebEngineView`` (``gui.sketcher_view``). Its files come from a
custom URL scheme, ``qb-sketcher://app/<file>``, answered from the editor's folder by a
``QWebEngineUrlSchemeHandler``, not from a local HTTP server: no port is opened, no other program can reach the
files, and the page's origin is QuantaBench's alone. Qt WebEngine accepts a custom scheme only when it is
registered before the ``QApplication`` exists, so ``gui/app.py`` calls ``prepare_before_app`` first.

Qt WebEngine is in ``PySide6-Addons`` up to PySide6 6.11 and in ``PySide6-WebEngine`` from 6.12; QuantaBench itself
needs only ``PySide6-Essentials``, so it is optional (``pip install "quantabench[editor]"``). Without it, or with
``QUANTABENCH_NO_SKETCHER=1`` (a machine whose GPU driver has trouble with Chromium), nothing of WebEngine is
loaded and the dock says what to install. This module never imports WebEngine at import time.
"""

from __future__ import annotations

import importlib.util
import os
import sys

SCHEME = b"qb-sketcher"
HOST = "app"
ENV_DISABLE = "QUANTABENCH_NO_SKETCHER"
ENV_GPU = "QUANTABENCH_SKETCHER_GPU"
INSTALL_HINT = ('pip install "quantabench[editor]" (from a checkout: pip install -e ".[editor]"), which adds Qt '
                "WebEngine (PySide6-WebEngine for PySide6 6.12 and newer, PySide6-Addons before), then restart "
                "QuantaBench")

_state = {"registered": False, "error": ""}


def webengine_installed() -> bool:
    """Qt WebEngine's Python modules are installed (not imported here)."""
    try:
        return (importlib.util.find_spec("PySide6.QtWebEngineWidgets") is not None
                and importlib.util.find_spec("PySide6.QtWebEngineCore") is not None)
    except (ImportError, ValueError):
        return False


def registered() -> bool:
    return _state["registered"]


def problem() -> str:
    """Why the sketcher cannot run in this process ('' = it can)."""
    if os.environ.get(ENV_DISABLE, "").strip() not in ("", "0"):
        return f"The sketcher is turned off on this machine ({ENV_DISABLE} is set)."
    if not webengine_installed():
        return "The sketcher needs Qt WebEngine, which is not installed. " + INSTALL_HINT + "."
    if not _state["registered"]:
        if _state["error"] and not _state["error"].startswith("the application started"):
            return f"Qt WebEngine could not be loaded ({_state['error']}). Reinstall it: {INSTALL_HINT}."
        return ("The sketcher's web page cannot be served in this session: its URL scheme must be registered before "
                "the application starts (gui/app.py does this). Restart QuantaBench.")
    return ""


def _sandbox_for_root() -> None:
    """Chromium refuses to start its sandbox as root ("Running as root without --no-sandbox is not supported"),
    which only happens in containers and CI on Linux. Only then is the sandbox turned off; a normal user keeps it."""
    if sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")


def _software_composition() -> None:
    """Draw the web view's frames with Qt Quick's software renderer (Chromium itself may still use the GPU).

    With Qt's default GPU composition, a QWebEngineView switches its whole window to OpenGL (RHI) composition,
    and that and VTK's OpenGL 3D view spoil each other: under Linux/Mesa the sketcher's window stayed black and
    the 3D view drew clipped, whether the sketcher was docked, floating or a window of its own. With the software
    renderer both draw correctly (checked under xvfb with pyvistaqt next to it), and a 2D editor does not need
    more. ``QUANTABENCH_SKETCHER_GPU=1`` keeps Qt's default instead (to try on a machine where it is faster)."""
    if os.environ.get(ENV_GPU, "").strip() not in ("", "0"):
        return
    try:
        from PySide6.QtQuick import QQuickWindow, QSGRendererInterface

        QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.Software)
    except ImportError:
        os.environ.setdefault("QT_QUICK_BACKEND", "software")


def prepare_before_app() -> bool:
    """Register the ``qb-sketcher`` scheme and Qt WebEngine's OpenGL context sharing. Must run before the
    ``QApplication`` is created; does nothing (returns False) when WebEngine is missing or turned off. Never raises:
    the rest of the app starts either way, and the Sketch dock reports ``problem()``."""
    if _state["registered"]:
        return True
    if os.environ.get(ENV_DISABLE, "").strip() not in ("", "0") or not webengine_installed():
        return False
    try:
        from PySide6.QtCore import QCoreApplication, Qt

        if QCoreApplication.instance() is not None:
            _state["error"] = "the application started before the sketcher's URL scheme could be registered"
            return False
        _sandbox_for_root()
        from PySide6.QtWebEngineCore import QWebEngineUrlScheme

        scheme = QWebEngineUrlScheme(SCHEME)
        scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)       # standard URLs: relative paths resolve
        flag = QWebEngineUrlScheme.Flag
        # SecureScheme: a secure context, like https. CorsEnabled + FetchApiAllowed: Emscripten fetch()es
        # Sketcher.wasm (without FetchApiAllowed Chromium answers 'URL scheme "qb-sketcher" is not supported').
        scheme.setFlags(flag.SecureScheme | flag.CorsEnabled | flag.FetchApiAllowed | flag.ContentSecurityPolicyIgnored)
        QWebEngineUrlScheme.registerScheme(scheme)
        # Qt WebEngine shares OpenGL contexts with the app; it must be set before the QApplication exists
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        _software_composition()
        import PySide6.QtWebEngineWidgets  # noqa: F401 - loaded now so WebEngine initializes with the app
    except Exception as exc:  # noqa: BLE001 - a broken WebEngine install must not stop the app
        _state["error"] = f"{type(exc).__name__}: {exc}"
        return False
    _state["registered"] = True
    _state["error"] = ""
    return True
