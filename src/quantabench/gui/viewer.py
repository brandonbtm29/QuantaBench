"""3D structure viewer (PyVista/VTK inside Qt).

Atoms are glyphed spheres and bonds are half-colored tubes, so a 3,000-atom slab renders as a
handful of meshes instead of thousands of actors. If OpenGL/VTK is unavailable (remote desktop,
headless CI) the widget shows a message instead of crashing the app.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii, vdw_radii
from ase.data.colors import jmol_colors
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

STYLES = ("Ball and stick", "Licorice", "Space filling", "Wireframe")


def bonds_from_rdkit(mol) -> np.ndarray:
    return np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int).reshape(-1, 2)


def bonds_from_distances(atoms: Atoms, scale: float = 1.15) -> np.ndarray:
    from ase.neighborlist import neighbor_list

    cut = [covalent_radii[z] * scale for z in atoms.numbers]
    i, j = neighbor_list("ij", atoms, cut, self_interaction=False)
    pairs = np.column_stack([i, j])
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    # drop bonds that wrap across periodic boundaries (drawn as long sticks otherwise)
    if len(pairs):
        d = np.linalg.norm(atoms.positions[pairs[:, 0]] - atoms.positions[pairs[:, 1]], axis=1)
        pairs = pairs[d < 3.0]
    return pairs


class MoleculeViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = None
        self.error = ""
        self.style = STYLES[0]
        self.show_hydrogens = True
        self.show_cell = True
        self._atoms: Atoms | None = None
        self._bonds = np.zeros((0, 2), int)
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor

            pv.global_theme.smooth_shading = True
            self.plotter = QtInteractor(self)
            self.plotter.enable_anti_aliasing("ssaa")
            lay.addWidget(self.plotter.interactor)
            self._set_background()
        except Exception as exc:  # no OpenGL, missing VTK, headless
            self.error = f"3D view unavailable: {type(exc).__name__}: {exc}"
            msg = QLabel(self.error)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setWordWrap(True)
            lay.addWidget(msg)

    # ------------------------------------------------------------------ public
    def show_atoms(self, atoms: Atoms, bonds: np.ndarray | None = None, reset_camera: bool = True) -> None:
        self._atoms = atoms
        self._bonds = bonds_from_distances(atoms) if bonds is None else bonds
        self._render(reset_camera)

    def show_rdkit(self, mol, conf_id: int = -1, reset_camera: bool = True) -> None:
        conf = mol.GetConformer(conf_id)
        atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=np.array(conf.GetPositions()))
        self.show_atoms(atoms, bonds_from_rdkit(mol), reset_camera)

    def set_style(self, style: str) -> None:
        self.style = style
        self._render(False)

    def set_show_hydrogens(self, on: bool) -> None:
        self.show_hydrogens = on
        self._render(False)

    def clear(self) -> None:
        self._atoms = None
        if self.plotter is not None:
            self.plotter.clear()
            self.plotter.render()

    def reset_camera(self) -> None:
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.render()

    def screenshot(self, path: str) -> None:
        if self.plotter is not None:
            self.plotter.screenshot(path, transparent_background=False)

    def close_viewer(self) -> None:
        if self.plotter is not None:
            self.plotter.close()

    # ------------------------------------------------------------------ drawing
    def _set_background(self) -> None:
        if self.plotter is None:
            return
        dark = self.palette().window().color().lightness() < 128
        self.plotter.set_background("#1e1f24" if dark else "#f4f5f7", top="#2b2d35" if dark else "#ffffff")

    def _render(self, reset_camera: bool) -> None:
        if self.plotter is None or self._atoms is None:
            return
        import pyvista as pv

        p = self.plotter
        p.clear()
        atoms = self._atoms
        keep = np.ones(len(atoms), bool) if self.show_hydrogens else atoms.numbers != 1
        idx_map = -np.ones(len(atoms), int)
        idx_map[np.where(keep)[0]] = np.arange(keep.sum())
        pos = atoms.positions[keep]
        z = atoms.numbers[keep]
        colors = (jmol_colors[z] * 255).astype(np.uint8)

        style = self.style
        if style == "Space filling":
            radii = np.nan_to_num(vdw_radii[z], nan=1.8)
        elif style == "Ball and stick":
            radii = 0.25 + 0.2 * covalent_radii[z]
        elif style == "Licorice":
            radii = np.full(len(z), 0.18)
        else:
            radii = np.zeros(len(z))

        if style != "Wireframe" and len(pos):
            cloud = pv.PolyData(pos)
            cloud["radius"] = radii
            cloud["rgb"] = colors
            spheres = cloud.glyph(scale="radius", geom=pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20),
                                  orient=False)
            p.add_mesh(spheres, scalars="rgb", rgb=True, specular=0.4, specular_power=20, name="atoms")

        bonds = self._bonds
        if len(bonds) and style != "Space filling":
            b = bonds[keep[bonds[:, 0]] & keep[bonds[:, 1]]]
            b = idx_map[b]
            if len(b):
                a, c = pos[b[:, 0]], pos[b[:, 1]]
                mid = (a + c) / 2
                pts = np.vstack([a, mid, mid, c])
                n = len(b)
                lines = np.hstack([np.column_stack([np.full(n, 2), np.arange(n), np.arange(n, 2 * n)]),
                                   np.column_stack([np.full(n, 2), np.arange(2 * n, 3 * n), np.arange(3 * n, 4 * n)])])
                seg_colors = np.vstack([colors[b[:, 0]], colors[b[:, 1]]])
                mesh = pv.PolyData(pts, lines=lines.ravel())
                mesh.cell_data["rgb"] = seg_colors
                if style == "Wireframe":
                    p.add_mesh(mesh, scalars="rgb", rgb=True, line_width=3, name="bonds")
                else:
                    tube_r = 0.18 if style == "Licorice" else 0.11
                    p.add_mesh(mesh.tube(radius=tube_r, n_sides=14), scalars="rgb", rgb=True, specular=0.3,
                               name="bonds")

        if self.show_cell and atoms.cell.rank == 3 and any(atoms.pbc):
            cell = atoms.cell.array
            corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) @ cell
            edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            lines = np.hstack([[2, i, j] for i, j in edges])
            p.add_mesh(pv.PolyData(corners, lines=lines), color="#8a8f98", line_width=1.5, name="cell")

        if reset_camera:
            p.reset_camera()
        p.render()


def element_counts(atoms: Atoms) -> str:
    from collections import Counter

    sym = {v: k for k, v in atomic_numbers.items()}
    c = Counter(atoms.numbers)
    return " ".join(f"{sym[z]}{n}" for z, n in sorted(c.items(), key=lambda t: (t[0] != 6, t[0] != 1, t[0])))
