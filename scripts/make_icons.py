"""Render the app icon (src/quantabench/assets/quantabench.svg) to the files the launchers use.

    python scripts/make_icons.py            # rewrites src/quantabench/assets/quantabench.{png,ico,icns}

- ``quantabench.png``: 256 x 256, for the window icon (Qt) and the Linux ``.desktop`` launcher.
- ``quantabench.ico``: 16 to 256 px (PNG entries, Windows Vista and later), for the Windows shortcuts.
- ``quantabench.icns``: 16 to 1024 px, for ``QuantaBench.app`` on macOS. macOS icons keep a margin around the
  rounded square (Apple's grid: 824 of 1024 px), so the shape is drawn smaller there.

Qt's SVG renderer draws the pictures (PySide6, already a dependency); the ICO and ICNS containers are written
here, so nothing else is needed. Run it again after changing the SVG and commit the four files.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "src" / "quantabench" / "assets"
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
#: ICNS entry types holding PNG data, by pixel size (Apple's ``iconutil`` writes the same ones)
ICNS_TYPES = ((b"icp4", 16), (b"ic11", 32), (b"icp5", 32), (b"ic12", 64), (b"ic07", 128), (b"ic13", 256),
              (b"ic08", 256), (b"ic14", 512), (b"ic09", 512), (b"ic10", 1024))
MAC_SHAPE = 824 / 1024          # the rounded square's share of a macOS icon's canvas


def render(svg: Path, size: int, shape: float = 1.0) -> bytes:
    """PNG bytes of ``svg`` drawn at ``size`` px, the drawing scaled to ``shape`` of the canvas and centered."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    _app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 (QPainter needs one)
    renderer = QSvgRenderer(str(svg))
    if not renderer.isValid():
        raise SystemExit(f"{svg} is not a valid SVG")
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    side = size * shape
    off = (size - side) / 2
    renderer.render(p, QRectF(off, off, side, side))
    p.end()
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(data)


def ico(images: dict[int, bytes]) -> bytes:
    """A Windows .ico holding one PNG per size."""
    sizes = sorted(images)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = len(header) + 16 * len(sizes)
    entries, blobs = [], []
    for s in sizes:
        blob = images[s]
        dim = 0 if s >= 256 else s                     # 0 means 256
        entries.append(struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset))
        blobs.append(blob)
        offset += len(blob)
    return header + b"".join(entries) + b"".join(blobs)


def icns(images: dict[int, bytes]) -> bytes:
    """A macOS .icns holding PNG entries (``ICNS_TYPES``)."""
    body = b"".join(kind + struct.pack(">I", len(images[s]) + 8) + images[s] for kind, s in ICNS_TYPES)
    return b"icns" + struct.pack(">I", len(body) + 8) + body


def main() -> int:
    svg = ASSETS / "quantabench.svg"
    full = {s: render(svg, s) for s in sorted({*ICO_SIZES, 256})}
    mac = {s: render(svg, s, MAC_SHAPE) for s in sorted({s for _k, s in ICNS_TYPES})}
    (ASSETS / "quantabench.png").write_bytes(full[256])
    (ASSETS / "quantabench.ico").write_bytes(ico({s: full[s] for s in ICO_SIZES}))
    (ASSETS / "quantabench.icns").write_bytes(icns(mac))
    for name in ("quantabench.png", "quantabench.ico", "quantabench.icns"):
        print(f"{ASSETS / name}: {(ASSETS / name).stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
