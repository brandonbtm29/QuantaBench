"""File > Import calculation output... (cclib): Gaussian, ORCA, Psi4, xtb, MOPAC, ... output files.

CONTRACT (implemented by the spectra/interop builder):

    install(win: MainWindow) -> None
        Adds the File menu action. Reads the chosen file with ``chem.io_cclib.read_output`` on a worker thread
        (``gui.workers.FunctionWorker``), turns it into a built ``MoleculeResult`` (one conformer, method from
        the file, charges, vibrations dict when the file has frequencies) and calls ``win.add_result(res)``.
"""

from __future__ import annotations


def install(win) -> None:
    """No-op until implemented."""
