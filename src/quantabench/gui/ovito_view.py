"""Optional OVITO viewport (``pip install ovito``), for large systems and trajectories.

OVITO's Python module embeds its interactive viewport as a PySide6 widget
(``ovito.gui.create_qwidget``, OVITO >= 3.12), which is why QuantaBench uses PySide6.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDockWidget


def ovito_dock(atoms, title: str, parent) -> QDockWidget:
    from ovito.gui import create_qwidget
    from ovito.io.ase import ase_to_ovito
    from ovito.pipeline import Pipeline, StaticSource

    data = ase_to_ovito(atoms)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.add_to_scene()
    window = create_qwidget(pipeline, parent=parent)
    widget = getattr(window, "widget", window)  # 3.16+ returns a WidgetViewportWindow
    viewport = getattr(window, "viewport", None)
    if viewport is not None:
        try:
            viewport.zoom_all()
        except Exception:  # noqa: S110
            pass
    dock = QDockWidget(title, parent)
    dock.setObjectName(f"ovito_{title}")
    dock.setWidget(widget)
    dock._ovito_refs = (pipeline, window)  # keep alive
    return dock
