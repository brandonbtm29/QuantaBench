"""Schrödinger's 2D Sketcher for the Sketch dock (UI-6): where its files are, and fetching them.

The Sketch dock runs the WebAssembly build of Schrödinger's Sketcher (https://github.com/schrodinger/sketcher,
BSD-3-Clause, built on RDKit and Qt 6) in Qt WebEngine (``gui.sketcher_web``). Its files are not in this
repository: the release zip is 17.5 MB and ``Sketcher.wasm`` alone 44.5 MB. They are downloaded on first use
(or ahead of time with ``quantabench-cli sketcher --install``) into the programs folder, the same per-user
folder as the macOS CREST build (``~/Library/Application Support/QuantaBench/programs`` on macOS,
``%LOCALAPPDATA%\\QuantaBench\\programs`` on Windows, ``~/.local/share/QuantaBench/programs`` on Linux), one folder
per release. The zip is pinned by sha256; a file that does not match is refused. Schrödinger's BSD-3 licence
text is written next to the files (``LICENSE.txt``; the repository keeps a copy in ``quantabench/licenses``).

Which folder is used, first match wins:

1. ``QUANTABENCH_SKETCHER_DIR``: a folder with ``wasm_shell.html``, ``qtloader.js``, ``Sketcher.js`` and
   ``Sketcher.wasm`` (a self-built or newer version).
2. This machine's ``sketcher`` entry in ``engines.machine_paths`` (``quantabench-cli sketcher --use DIR``; the
   settings file is shared by every machine, so the path is kept per machine).
3. The pinned release in the programs folder (``schrodinger-sketcher-2026-4-wasm``).

Offline: download ``URL`` by hand, then ``quantabench-cli sketcher --install --from-zip FILE`` (or "Install
from a zip file..." in the dock), or put the zip into the programs folder; it is checked and unpacked the next
time the dock opens. Nothing here needs Qt.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from platformdirs import user_data_dir

APP_NAME = "QuantaBench"
RELEASE = "v2026-4"
ZIP_NAME = "sketcher-2026-4-wasm.zip"
URL = f"https://github.com/schrodinger/sketcher/releases/download/{RELEASE}/{ZIP_NAME}"
SHA256 = "eb87b1850c6526b7833880a9d7fa7dcbebf0c723d572c3c907d7ccb58bacdae8"
SIZE = 17_541_459
FOLDER = "schrodinger-sketcher-2026-4-wasm"     # under the programs folder
PROJECT_URL = "https://github.com/schrodinger/sketcher"
#: the files the dock loads (the zip also has icons and smiles_to_helm.js/.wasm, which are unpacked too)
REQUIRED = ("wasm_shell.html", "qtloader.js", "Sketcher.js", "Sketcher.wasm")
START_PAGE = "wasm_shell.html"
ENV_DIR = "QUANTABENCH_SKETCHER_DIR"
ENGINE_KEY = "sketcher"                          # engines.machine_paths entry (quantabench-cli sketcher --use)
LICENSE_FILE = "LICENSE.txt"
SOURCE_FILE = "SOURCE.txt"
LICENSE_COPY = Path(__file__).resolve().parent / "licenses" / "schrodinger-sketcher-LICENSE.txt"
_CHUNK = 1 << 20
_MAX_UNPACKED = 512 * 1024 * 1024                # sanity limit for a zip given by hand (the release is 57 MB)


class SketcherAssetError(RuntimeError):
    """The editor's files could not be found, downloaded, checked or unpacked. The message says what to do."""


@dataclass
class Location:
    """Where the editor's files are (``path``, None when they are missing) and why that folder."""
    path: Path | None
    source: str          # "environment" | "settings" | "download" | "missing"
    problem: str = ""    # why a configured folder cannot be used, or how to get the files

    @property
    def ok(self) -> bool:
        return self.path is not None


def programs_dir() -> Path:
    """The per-user folder for downloaded programs (also where install_macos.sh puts the CREST build)."""
    return Path(user_data_dir(APP_NAME, appauthor=False)) / "programs"


def release_dir(programs: Path | None = None) -> Path:
    return (programs or programs_dir()) / FOLDER


def missing_files(folder: Path) -> list[str]:
    """The files of ``REQUIRED`` that ``folder`` lacks (an empty ``Sketcher.wasm`` counts as missing)."""
    out = []
    for name in REQUIRED:
        p = folder / name
        if not p.is_file() or p.stat().st_size == 0:
            out.append(name)
    return out


def is_complete(folder: Path) -> bool:
    return folder.is_dir() and not missing_files(folder)


def _configured(settings) -> str:
    """This machine's ``sketcher`` path from the settings ('' = none)."""
    if settings is None:
        try:
            from . import settings as settings_mod

            settings = settings_mod.load()
        except Exception:  # noqa: BLE001 - unreadable settings: use the download
            return ""
    try:
        return settings.engine_paths_for().get(ENGINE_KEY, ("", ""))[0]
    except Exception:  # noqa: BLE001
        return ""


def locate(settings=None, programs: Path | None = None) -> Location:
    """Where the editor's files are: ``QUANTABENCH_SKETCHER_DIR``, this machine's settings entry, then the pinned
    release in the programs folder. A configured folder without the files is reported, not skipped, so a typo
    is not hidden by a download."""
    for value, source, how in ((os.environ.get(ENV_DIR, "").strip(), "environment", f"{ENV_DIR}"),
                               (_configured(settings), "settings",
                                "this machine's sketcher folder (quantabench-cli sketcher --use)")):
        if not value:
            continue
        folder = Path(os.path.expanduser(value))
        missing = missing_files(folder) if folder.is_dir() else list(REQUIRED)
        if missing:
            return Location(None, source, f"{how} is {folder}, which lacks {', '.join(missing)}")
        return Location(folder, source)
    folder = release_dir(programs)
    if is_complete(folder):
        return Location(folder, "download")
    return Location(None, "missing", not_installed_message(programs))


def not_installed_message(programs: Path | None = None) -> str:
    return (f"Schrödinger's Sketcher {RELEASE} is not downloaded yet ({SIZE / 1e6:.1f} MB, BSD-3-Clause, "
            f"from {PROJECT_URL}). It goes into {release_dir(programs)}.")


def manual_instructions(programs: Path | None = None) -> str:
    """How to install the editor without QuantaBench's download (offline, or behind a proxy)."""
    folder = programs or programs_dir()
    return (f"Download it by hand:\n  {URL}\n  ({SIZE:,} bytes, sha256 {SHA256})\n"
            f"then run\n  quantabench-cli sketcher --install --from-zip <the downloaded zip>\n"
            f"or choose \"Install from a zip file...\" in the Sketch dock, or put the zip into\n  {folder}\n"
            "and open the Sketch dock again.")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, sha256: str | None = None) -> None:
    """Raise ``SketcherAssetError`` unless ``path`` has the pinned sha256 (``SHA256`` unless given)."""
    want = (sha256 or SHA256).lower()
    got = sha256_of(path)
    if got != want:
        raise SketcherAssetError(f"{path.name} is not the expected file: its sha256 is {got}, the pinned "
                                 f"release {RELEASE} has {want}. Download it again from {URL}.")


def download(dest: Path, url: str | None = None, sha256: str | None = None, size: int | None = None,
             progress: Callable[[int, int], None] | None = None, cancel: threading.Event | None = None,
             timeout: float = 60.0, opener=None) -> Path:
    """Download ``url`` (``URL``) to ``dest`` (through ``dest.part``), checking the sha256 (``SHA256``) as it
    arrives. Raises ``SketcherAssetError`` (with the manual instructions) when the network fails or the file does
    not match; nothing is left behind then. ``progress(done, total)`` is called from this thread."""
    url, sha256, size = url or URL, sha256 or SHA256, size or SIZE
    opener = opener or urllib.request.urlopen
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    done = 0
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "QuantaBench (sketcher download)"})
        with opener(req, timeout=timeout) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0) or size
            while True:
                if cancel is not None and cancel.is_set():
                    raise SketcherAssetError("Download cancelled.")
                block = resp.read(_CHUNK)
                if not block:
                    break
                out.write(block)
                h.update(block)
                done += len(block)
                if progress is not None:
                    progress(done, total)
    except SketcherAssetError:
        part.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:     # URLError, HTTPError, timeouts, a full disk
        part.unlink(missing_ok=True)
        raise SketcherAssetError(f"Could not download Schrödinger's Sketcher ({type(exc).__name__}: {exc}).\n"
                                 + manual_instructions(dest.parent)) from exc
    if h.hexdigest() != sha256.lower():
        part.unlink(missing_ok=True)
        raise SketcherAssetError(f"The download from {url} does not have the pinned sha256 ({done:,} bytes, sha256 "
                                 f"{h.hexdigest()}; expected {sha256.lower()}). Nothing was installed.")
    os.replace(part, dest)
    return dest


def _safe_members(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """The zip's files with their relative paths; refuses absolute paths, '..' and oversized archives."""
    out, total = [], 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        rel = PurePosixPath(info.filename.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or ":" in rel.parts[0]:
            raise SketcherAssetError(f"refusing to unpack {info.filename!r}: not a relative path inside the zip")
        total += info.file_size
        out.append((info, rel))
    if total > _MAX_UNPACKED:
        raise SketcherAssetError(f"the zip unpacks to {total / 1e6:.0f} MB, more than the editor can need")
    return out


def unpack(zip_path: Path, dest: Path) -> Path:
    """Unpack ``zip_path`` into ``dest`` (replacing it) through a temporary folder next to it, so a failure never
    leaves half a folder. A zip with one top folder holding the files is unpacked from inside that folder.
    Writes ``LICENSE.txt`` (Schrödinger's BSD-3 text) and ``SOURCE.txt`` (where the files came from)."""
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = _safe_members(zf)
            tops = {rel.parts[0] for _info, rel in members}
            strip = len(tops) == 1 and all(len(rel.parts) > 1 for _info, rel in members)
            for info, rel in members:
                target = tmp.joinpath(*(rel.parts[1:] if strip else rel.parts))
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, _CHUNK)
        missing = missing_files(tmp)
        if missing:
            raise SketcherAssetError(f"{zip_path.name} does not contain {', '.join(missing)}")
        write_notices(tmp, zip_path)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except zipfile.BadZipFile as exc:
        raise SketcherAssetError(f"{zip_path.name} is not a valid zip file ({exc})") from exc
    except OSError as exc:
        raise SketcherAssetError(f"could not unpack {zip_path.name} into {dest} ({exc})") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def write_notices(folder: Path, zip_path: Path | None = None) -> None:
    """``LICENSE.txt`` (Schrödinger's BSD-3-Clause licence) and ``SOURCE.txt`` next to the editor's files."""
    text = LICENSE_COPY.read_text(encoding="utf-8") if LICENSE_COPY.is_file() else (
        f"BSD 3-Clause License, Copyright (c) 2024, Schrodinger, LLC. Full text: {PROJECT_URL}/blob/main/LICENSE\n")
    (folder / LICENSE_FILE).write_text(text, encoding="utf-8")
    (folder / SOURCE_FILE).write_text(
        f"Schrödinger's 2D Sketcher, WebAssembly build, release {RELEASE}\n"
        f"Project: {PROJECT_URL} (BSD-3-Clause, see {LICENSE_FILE})\n"
        f"Downloaded from: {URL}\nsha256 of the zip: {SHA256}\n"
        + (f"Unpacked from: {zip_path.name}\n" if zip_path is not None else "")
        + "The bundle also contains Qt for WebAssembly files under their own licences (qtloader.js: Qt's "
          "commercial licence or GPL-3.0; the Qt libraries linked into Sketcher.wasm) and RDKit (BSD-3-Clause).\n"
        "Installed by QuantaBench for its Sketch dock. Delete this folder to remove it.\n", encoding="utf-8")


def install(programs: Path | None = None, zip_path: str | Path | None = None, force: bool = False,
            progress: Callable[[int, int], None] | None = None, cancel: threading.Event | None = None,
            url: str | None = None, sha256: str | None = None, size: int | None = None, opener=None,
            log: Callable[[str], None] | None = None) -> Path:
    """Make the pinned release ready in the programs folder and return its folder.

    Already unpacked (and not ``force``): nothing to do. Otherwise the zip is ``zip_path`` (a file downloaded by
    hand), else ``<programs>/sketcher-2026-4-wasm.zip`` when someone put it there, else it is downloaded. The
    zip must have the pinned sha256. A downloaded zip is deleted once unpacked; a zip given by hand is kept.
    ``url``/``sha256``/``size`` default to the pinned release (``URL``, ``SHA256``, ``SIZE``)."""
    url, sha256, size = url or URL, sha256 or SHA256, size or SIZE
    say = log or (lambda _msg: None)
    programs = programs or programs_dir()
    dest = release_dir(programs)
    if is_complete(dest) and not force:
        say(f"Schrödinger's Sketcher {RELEASE} is already in {dest}")
        return dest
    downloaded = False
    if zip_path is not None:
        source = Path(zip_path).expanduser()
        if not source.is_file():
            raise SketcherAssetError(f"{source} does not exist")
    elif (programs / ZIP_NAME).is_file():
        source = programs / ZIP_NAME
        say(f"Using {source}")
    else:
        say(f"Downloading {url} ({size / 1e6:.1f} MB)...")
        source = download(programs / ZIP_NAME, url=url, sha256=sha256, size=size, progress=progress, cancel=cancel,
                          opener=opener)
        downloaded = True
    if not downloaded:
        verify(source, sha256)
    unpack(source, dest)
    if downloaded:
        source.unlink(missing_ok=True)
    say(f"Schrödinger's Sketcher {RELEASE} is ready in {dest}")
    return dest


def status_text(settings=None, programs: Path | None = None) -> str:
    """One paragraph for ``quantabench-cli sketcher`` and ``doctor``."""
    loc = locate(settings, programs)
    if loc.ok:
        where = {"environment": f"from {ENV_DIR}", "settings": "this machine's folder (quantabench-cli sketcher --use)",
                 "download": f"release {RELEASE}"}[loc.source]
        return f"Schrödinger's Sketcher: {loc.path} ({where})"
    if loc.source in ("environment", "settings"):
        return f"Schrödinger's Sketcher: not usable: {loc.problem}"
    return f"Schrödinger's Sketcher: not downloaded (quantabench-cli sketcher --install, {SIZE / 1e6:.1f} MB)"
