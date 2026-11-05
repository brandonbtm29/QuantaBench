"""Machines dock: this computer's lanes and the other machines in the registry (workers).

CONTRACT (implemented in integration after the dist builders land):

    install(win: MainWindow) -> None
        "Machines" dock (right area, tabified with Pipeline): the machine registry (dist.profile), each
        worker's status (dist.client), the lane plan for the current settings (dist.lanes), and a
        "Run on" choice the Run buttons use.
"""

from __future__ import annotations


def install(win) -> None:
    """No-op until implemented."""
