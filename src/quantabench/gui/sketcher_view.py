"""Schrödinger's Sketcher in a ``QWebEngineView``, and the part of its JavaScript API the Sketch dock uses.

Imports Qt WebEngine at import time: import it only when ``gui.sketcher_web.problem()`` is empty.

Serving (``SketcherSchemeHandler``): ``qb-sketcher://app/<file>`` is answered from the editor's folder
(``sketcher_assets.locate``), only for files directly inside it, with these content types: ``Sketcher.wasm`` as
``application/wasm``, so ``WebAssembly.instantiateStreaming`` compiles it while it arrives (3 to 5 s for the
44.5 MB module under xvfb). No cross-origin isolation headers (COOP/COEP) are sent: this build uses no threads
(``Sketcher.js`` has no SharedArrayBuffer or pthreads), so it does not need them. The handler sits on Qt's default
profile, which is off the record in Qt 6 (nothing written to disk).

The JavaScript API (from Schrödinger's React wrapper, ``wasm/package/src/Sketcher.tsx``): ``wasm_shell.html``
stores the instance as ``window.Module`` once ``qtLoad`` resolves; ``Module.sketcher_import_text(text)``
(format detected; without ``sketcher_clear`` it adds the structure next to the drawing),
``Module.sketcher_export_text(Module.Format.X)``, ``sketcher_clear()``, ``sketcher_is_empty()``; C++ exceptions
arrive as numbers, turned into text with ``Module.getExceptionMessage``. ``Module.sketcher_changed_callback`` is
called (100 ms after) every change: it logs a marker that ``SketcherPage.javaScriptConsoleMessage`` turns into the
``changed`` signal, which needs no QWebChannel.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from PySide6.QtCore import QCoreApplication, QFile, QIODevice, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineCore import (
    QWebEnginePage,
    QWebEngineProfile,
    QWebEngineSettings,
    QWebEngineUrlRequestJob,
    QWebEngineUrlSchemeHandler,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from ..sketcher_assets import START_PAGE
from .sketcher_web import HOST, SCHEME

SCHEME_NAME = SCHEME.decode()
START_URL = f"{SCHEME_NAME}://{HOST}/{START_PAGE}"
LOAD_TIMEOUT_S = 180.0          # compiling 44.5 MB of WebAssembly on a slow machine without a GPU
POLL_MS = 250
CHANGED_MARK = "qb-sketcher:changed"
#: export formats of ``Module.Format`` the dock may ask for (a whitelist: the name goes into JavaScript)
FORMATS = ("SMILES", "EXTENDED_SMILES", "MDL_MOLV2000", "MDL_MOLV3000", "INCHI", "INCHI_KEY", "CDXML", "MRV",
           "RDMOL_BINARY_BASE64")
MIME_TYPES = {".html": b"text/html", ".js": b"text/javascript", ".mjs": b"text/javascript",
              ".wasm": b"application/wasm", ".json": b"application/json", ".css": b"text/css",
              ".svg": b"image/svg+xml", ".png": b"image/png", ".gif": b"image/gif", ".ico": b"image/x-icon",
              ".txt": b"text/plain"}

# The scripts answer with JSON text: PySide6 6.12 hands JavaScript objects and arrays from runJavaScript back as ''
READY_JS = """(function () {
  var t = document.getElementById('error-text');
  if (t && t.textContent) { return JSON.stringify({state: 'error', text: t.textContent}); }
  if (window.Module && typeof window.Module.sketcher_export_text === 'function') {
    return JSON.stringify({state: 'ready'});
  }
  var s = document.getElementById('qtstatus');
  return JSON.stringify({state: 'loading', text: s ? s.textContent : ''});
})()"""
HOOK_JS = ("Module.sketcher_changed_callback = function () { console.log(" + json.dumps(CHANGED_MARK)
           + "); }; return true;")
#: runs a function body; answers {ok, value} or {ok: false, error} with the C++ exception's message, as JSON
CALL_JS = """(function () {
  try { return JSON.stringify({ok: true, value: (function () { %s })()}); }
  catch (e) {
    var m = e;
    try {
      if (window.Module && typeof Module.getExceptionMessage === 'function' && (typeof e === 'number'
          || (typeof WebAssembly.Exception === 'function' && e instanceof WebAssembly.Exception))) {
        m = Module.getExceptionMessage(e);
      }
    } catch (e2) {}
    if (Array.isArray(m)) { m = m.filter(function (x) { return !!x; }).pop() || 'unknown error'; }
    return JSON.stringify({ok: false, error: String(m && m.message ? m.message : m)});
  }
})()"""


def _decode(answer) -> dict:
    """A script's JSON answer as a dict ({} when there is none: the page was navigating, or the script failed)."""
    if isinstance(answer, dict):
        return answer
    try:
        value = json.loads(answer) if isinstance(answer, str) and answer else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class SketcherSchemeHandler(QWebEngineUrlSchemeHandler):
    """Answers ``qb-sketcher://app/<file>`` with ``<root>/<file>``; anything else fails."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root: Path | None = None

    def resolve(self, url: QUrl) -> Path | None:
        """The file for ``url``, or None (wrong host, outside the folder, missing)."""
        if self.root is None or url.host() != HOST:
            return None
        rel = PurePosixPath(url.path().lstrip("/") or START_PAGE)
        if ".." in rel.parts or not rel.parts:
            return None
        path = self.root.joinpath(*rel.parts)
        try:
            path.resolve().relative_to(self.root.resolve())
        except (ValueError, OSError):
            return None
        return path if path.is_file() else None

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802 - Qt's name
        if bytes(job.requestMethod().data()) not in (b"GET", b"HEAD"):
            job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
            return
        path = self.resolve(job.requestUrl())
        if path is None:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        f = QFile(str(path), job)              # owned by the job: closed when Qt is done with it
        if not f.open(QIODevice.OpenModeFlag.ReadOnly):
            job.fail(QWebEngineUrlRequestJob.Error.RequestFailed)
            return
        job.reply(MIME_TYPES.get(path.suffix.lower(), b"application/octet-stream"), f)


_HANDLER: list[SketcherSchemeHandler] = []


def scheme_handler(root: Path) -> SketcherSchemeHandler:
    """The scheme handler (one per process, on the default profile), now serving ``root``."""
    if not _HANDLER:
        handler = SketcherSchemeHandler(QCoreApplication.instance())
        QWebEngineProfile.defaultProfile().installUrlSchemeHandler(SCHEME, handler)
        _HANDLER.append(handler)
    _HANDLER[0].root = Path(root)
    return _HANDLER[0]


class SketcherPage(QWebEnginePage):
    """The sketcher's page: change marker → ``changed``, other console lines → ``console``; it never navigates
    away (links open in the system browser); clipboard access allowed for the sketcher's own origin."""

    changed = Signal()
    console = Signal(str)

    def __init__(self, profile, parent=None):
        super().__init__(profile, parent)
        attrs = self.settings()
        attrs.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        attrs.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanPaste, True)
        if hasattr(self, "permissionRequested"):          # Qt 6.8+
            self.permissionRequested.connect(self._on_permission)

    def _on_permission(self, permission) -> None:
        from PySide6.QtWebEngineCore import QWebEnginePermission

        ours = permission.origin().scheme() == SCHEME_NAME
        if ours and permission.permissionType() == QWebEnginePermission.PermissionType.ClipboardReadWrite:
            permission.grant()
        else:
            permission.deny()

    def javaScriptConsoleMessage(self, level, message, line, source) -> None:  # noqa: N802 - Qt's name
        if message == CHANGED_MARK:
            self.changed.emit()
            return
        self.console.emit(f"{message} ({source}:{line})" if source else message)

    def acceptNavigationRequest(self, url, nav_type, is_main_frame) -> bool:  # noqa: N802 - Qt's name
        if url.scheme() == SCHEME_NAME or (not is_main_frame and url.scheme() in ("about", "data", "blob")):
            return True
        if url.scheme() in ("http", "https", "mailto") and \
                nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url)
        return False


class SketcherView(QWebEngineView):
    """The editor. ``start(root)`` loads it; ``ready(seconds)`` once ``window.Module`` answers, or
    ``failed(message)``. ``changed`` follows every edit. The ``import_text``/``export_text``/... methods call the
    JavaScript API; their callbacks get ``(ok, value_or_error_message)`` on the GUI thread."""

    ready = Signal(float)
    failed = Signal(str)
    changed = Signal()
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPage(SketcherPage(QWebEngineProfile.defaultProfile(), self))   # deleted with the view
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)        # the sketcher has its own menus
        self.page().changed.connect(self.changed)
        self.page().console.connect(self._on_console)
        self.page().renderProcessTerminated.connect(self._on_terminated)
        self.loadFinished.connect(self._on_load_finished)
        self.is_ready = False
        self.load_seconds = 0.0
        self.last_console = ""
        self._t0 = 0.0
        self._poll_sent = 0.0           # when the last readiness poll went out (0 = none in flight)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    # -- loading
    def start(self, root: Path) -> None:
        scheme_handler(root)
        self.is_ready = False
        self._t0 = time.monotonic()
        self._poll_sent = 0.0
        self.status.emit("Loading the sketcher...")
        self.load(QUrl(START_URL))
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _on_console(self, text: str) -> None:
        self.last_console = text

    def _fail(self, message: str) -> None:
        self._timer.stop()
        self.is_ready = False
        self.failed.emit(message)

    def _on_load_finished(self, ok: bool) -> None:
        if not ok and not self.is_ready and self._timer.isActive():
            self._fail(f"The sketcher's page ({START_URL}) did not load."
                       + (f" Last message: {self.last_console}" if self.last_console else ""))

    def _on_terminated(self, status, code) -> None:
        name = getattr(status, "name", str(status))
        self._fail(f"The sketcher's web process stopped ({name}, exit code {code}). Press Reload to start it again.")

    def _poll(self) -> None:
        # a script sent while the page navigates gets no answer: ask again after a while instead of waiting
        if self._poll_sent and time.monotonic() - self._poll_sent < 2.0:
            return
        if time.monotonic() - self._t0 > LOAD_TIMEOUT_S:
            self._fail(f"The sketcher did not start within {LOAD_TIMEOUT_S:.0f} s."
                       + (f" Last message: {self.last_console}" if self.last_console else ""))
            return
        self._poll_sent = time.monotonic()
        self.page().runJavaScript(READY_JS, 0, self._on_poll)

    def _on_poll(self, answer) -> None:
        self._poll_sent = 0.0
        if not self._timer.isActive():
            return
        value = _decode(answer)
        state = value.get("state", "")
        if state == "error":
            self._fail(f"The sketcher could not start: {value.get('text') or 'unknown error'}")
        elif state == "ready":
            self._timer.stop()
            self._call(HOOK_JS, self._on_hooked)
        elif value.get("text"):
            self.status.emit(f"Loading the sketcher: {value['text']}")

    def _on_hooked(self, ok: bool, value) -> None:
        if not ok:
            self._fail(f"The sketcher started but its API did not answer ({value}).")
            return
        self.is_ready = True
        self.load_seconds = time.monotonic() - self._t0
        self.ready.emit(self.load_seconds)

    # -- the JavaScript API
    def _call(self, body: str, callback: Callable[[bool, object], None] | None = None) -> None:
        def done(answer) -> None:
            if callback is None:
                return
            result = _decode(answer)
            if "ok" in result:
                callback(bool(result["ok"]), result.get("value") if result["ok"] else result.get("error", ""))
            else:
                callback(False, "the sketcher did not answer")
        self.page().runJavaScript(CALL_JS % body, 0, done)

    def import_text(self, text: str, clear: bool = False, callback=None) -> None:
        """Load ``text`` (SMILES, MOL block, ...: Sketcher detects the format); ``clear`` first, else it is added
        next to the drawing."""
        self._call(("Module.sketcher_clear(); " if clear else "")
                   + f"Module.sketcher_import_text({json.dumps(text)}); return true;", callback)

    def export_text(self, fmt: str, callback) -> None:
        if fmt not in FORMATS:
            raise ValueError(f"unknown sketcher format {fmt!r} (one of {', '.join(FORMATS)})")
        self._call(f"return Module.sketcher_export_text(Module.Format.{fmt});", callback)

    def clear(self, callback=None) -> None:
        self._call("Module.sketcher_clear(); return true;", callback)

    def is_empty(self, callback) -> None:
        self._call("return Module.sketcher_is_empty();", callback)
