"""Help > Check for updates...: pull the newest QuantaBench and restart.

CONTRACT:

    install(win: MainWindow) -> None
        Adds "Check for updates..." to the Help menu (before About). The work is in ``quantabench.update``
        (shared with ``quantabench-cli update``) and runs on a worker thread (``gui.workers.FunctionWorker``).
        The result is shown in a message box; when the checkout can be fast-forwarded it offers
        "Update and restart", which pulls, reinstalls if the dependencies changed, closes the window (the
        usual close: running work is cancelled first) and starts a new GUI process (``update.relaunch``).
"""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMessageBox

TITLE = "Check for updates"


def _help_menu(win):
    for a in win.menuBar().actions():
        if a.menu() is not None and a.text().replace("&", "") == "Help":
            return a.menu()
    return win.menuBar().addMenu("&Help")


def install(win) -> None:
    action = QAction("Check for updates...", win)
    action.setToolTip("Fetch the newest QuantaBench from GitHub (git checkout), update and restart")
    action.triggered.connect(lambda _checked=False: check_for_updates(win))
    menu = _help_menu(win)
    actions = menu.actions()
    if actions:                                   # before "About QuantaBench"
        menu.insertAction(actions[0], action)
        menu.insertSeparator(actions[0])
    else:
        menu.addAction(action)
    win.a_check_updates = action


def _busy(win, text: str | None) -> None:
    action = getattr(win, "a_check_updates", None)
    if action is not None:
        action.setEnabled(text is None)
    if text:
        win.statusBar().showMessage(text)
    else:
        win.statusBar().clearMessage()


def check_for_updates(win, repo=None) -> None:
    """Fetch on a worker thread, then show what was found."""
    from .. import update
    from . import workers

    _busy(win, "Checking for updates...")

    def done(status):
        _busy(win, None)
        show_status(win, status, repo)

    def failed(msg):
        _busy(win, None)
        QMessageBox.warning(win, TITLE, msg)

    workers.start(workers.FunctionWorker(update.check, repo), win, on_done=done, on_failed=failed)


def _ask_update(win, status) -> bool:
    """The result box; True when the user chose "Update and restart"."""
    box = QMessageBox(win)
    box.setWindowTitle(TITLE)
    box.setIcon(QMessageBox.Icon.Information if status.state != "error" else QMessageBox.Icon.Warning)
    box.setText(status.summary())
    update_btn = None
    if status.can_update:
        update_btn = box.addButton("Update and restart", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(update_btn)
    else:
        box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    return update_btn is not None and box.clickedButton() is update_btn


def show_status(win, status, repo=None) -> None:
    if not _ask_update(win, status):
        return
    if getattr(win, "worker", None) is not None:     # new code on disk under a running build mixes versions
        QMessageBox.information(win, TITLE, "A build is running. Let it finish (or cancel it), then update.")
        return
    run_update(win, repo)


def run_update(win, repo=None) -> None:
    """Pull and reinstall on a worker thread, then restart (or explain what went wrong)."""
    from .. import update
    from . import workers

    _busy(win, "Updating QuantaBench (git pull, then pip if the dependencies changed)...")

    def done(res):
        _busy(win, None)
        if hasattr(win, "log"):
            for line in res.message.splitlines():
                win.log(line)
        if res.ok and res.restart_needed:
            restart(win)
            return
        if res.restart_needed:                  # the code changed but pip failed
            answer = QMessageBox.warning(win, TITLE, res.message + "\n\nRestart anyway?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                restart(win)
            return
        (QMessageBox.information if res.ok else QMessageBox.warning)(win, TITLE, res.message)

    def failed(msg):
        _busy(win, None)
        QMessageBox.warning(win, TITLE, msg)

    workers.start(workers.FunctionWorker(update.update, repo), win, on_done=done, on_failed=failed)


def restart(win) -> bool:
    """Close the window, then start a new QuantaBench once the app quits.

    The window may refuse (the user keeps a running build) or wait for running work to stop; the new process
    starts only when the app really quits. Returns False when the window stayed open."""
    from .. import update

    started = []

    def relaunch():
        if not started:
            started.append(update.relaunch())

    win.close()
    if getattr(win, "closed", False) or not win.isVisible():
        relaunch()
        _quit_app()
        return True
    if getattr(win, "_quit_pending", False):    # closes by itself once the running work has stopped
        QApplication.instance().aboutToQuit.connect(relaunch)
        return True
    win.statusBar().showMessage("QuantaBench was updated: restart it to use the new version.")
    return False


def _quit_app() -> None:
    QApplication.instance().quit()
