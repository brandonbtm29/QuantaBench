"""Help > Check for updates... (pull the newest QuantaBench, then offer to restart) and Help > Create app shortcuts...

CONTRACT:

    install(win: MainWindow) -> None
        Adds "Check for updates..." and "Create app shortcuts..." to the Help menu (before About). The work is in
        ``quantabench.update`` (shared with ``quantabench-cli update``), ``quantabench.local_worker`` and
        ``quantabench.launchers`` (``quantabench-cli shortcuts``), and runs on worker threads
        (``gui.workers.FunctionWorker``).

    The update, step by step:
    1. ``update.check`` on a worker thread: the new commits on origin/main, the CHANGELOG.md sections they bring,
       the newest release, and the worker on this computer.
    2. ``UpdateDialog`` shows them (What's new rendered from the CHANGELOG, then the commits), says why updates
       follow main (``update.FOLLOWS``), and why an update is refused (local changes, another branch, own
       commits) or has to wait (the worker here runs jobs from this checkout). Buttons: Update / Later; "Update
       when idle" when only the worker is busy; Upgrade packages... / Close with nothing to pull. "Also upgrade
       the installed packages" adds pip's upgrade.
    3. Work running in this window (a build, local jobs in the queue) or a busy worker: the update waits for
       them ("Update when they finish", checked every ``WAIT_MS``; Help > Check for updates... offers to cancel
       the wait) or is cancelled. Jobs on other machines go on there and do not hold the update.
    4. ``update.update`` on a worker thread (fast-forward only; pip into this environment and the checkout's
       others when the dependencies changed; back to the old commit if pip fails). Then the result: what changed
       and "Restart now" / "Later". When the worker here runs this checkout, is idle and a service keeps it
       (LaunchAgent, scheduled task), "Restart this computer's worker too" restarts it
       (``local_worker.restart``); otherwise the message says how.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QTextBrowser,
    QVBoxLayout,
)

from . import keymap

TITLE = "Check for updates"
UPGRADE_TEXT = "Also upgrade the installed packages (pip; PyTorch and the fork packages are kept)"
UPGRADE_TIP = ("pip upgrades every package QuantaBench uses to the newest version pyproject.toml allows "
               "(quantabench-cli update --upgrade-packages). PyTorch stays the build you installed, and the "
               "forks listed in scripts/fork-packages.json stay too. This takes a few minutes, and the window "
               "lists what changed before it restarts. On Windows, close other QuantaBench windows and workers "
               "first: Windows keeps the files of running programs locked.")
SHOWN_CHANGES = 20                                 # package changes in the result box (the rest under Details)
WAIT_MS = 5000                                     # how often a waiting update looks again
WORKER_EVERY = 6                                   # ... and the worker on this computer every this many looks
SHORTCUTS_TITLE = "Create app shortcuts"


def _help_menu(win):
    for a in win.menuBar().actions():
        if a.menu() is not None and a.text().replace("&", "") == "Help":
            return a.menu()
    return win.menuBar().addMenu("&Help")


def install(win) -> None:
    action = QAction("Check for updates...", win)
    keymap.bind(win, action, "help.check_updates",
                "Fetch the newest QuantaBench from GitHub (git checkout), update and restart")
    action.triggered.connect(lambda _checked=False: check_for_updates(win))
    shortcuts = QAction("Create app shortcuts...", win)
    from .. import launchers

    keymap.bind(win, shortcuts, "help.create_shortcuts",
                f"Make {launchers.what()} that start QuantaBench without a terminal window, with its icon")
    shortcuts.triggered.connect(lambda _checked=False: create_shortcuts(win))
    menu = _help_menu(win)
    actions = menu.actions()
    if actions:                                   # before "About QuantaBench"
        menu.insertAction(actions[0], action)
        menu.insertAction(actions[0], shortcuts)
        menu.insertSeparator(actions[0])
    else:
        menu.addAction(action)
        menu.addAction(shortcuts)
    win.a_check_updates = action
    win.a_create_shortcuts = shortcuts


def _busy(win, text: str | None) -> None:
    action = getattr(win, "a_check_updates", None)
    if action is not None:
        action.setEnabled(text is None)
    if text:
        win.statusBar().showMessage(text)
    else:
        win.statusBar().clearMessage()


def check_for_updates(win, repo=None) -> None:
    """Fetch on a worker thread, then show what was found. While an update waits for running work, offer to
    cancel the wait instead."""
    from .. import update
    from . import workers

    waiter = getattr(win, "_update_wait", None)
    if waiter is not None:
        answer = QMessageBox.question(win, TITLE, f"An update waits: {waiter.why()}\n\nCancel the waiting update?")
        if answer == QMessageBox.StandardButton.Yes:
            waiter.cancel()
        return
    _busy(win, "Checking for updates...")

    def done(status):
        _busy(win, None)
        show_status(win, status, repo)

    def failed(msg):
        _busy(win, None)
        QMessageBox.warning(win, TITLE, msg)

    workers.start(workers.FunctionWorker(update.check, repo), win, on_done=done, on_failed=failed)


# --------------------------------------------------------------------------------------------- the dialog
class UpdateDialog(QDialog):
    """What a check found; ``choice`` after ``exec``: False, "update" or "upgrade"."""

    def __init__(self, status, parent=None):
        from .. import update

        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.choice: str | bool = False
        self.status = status
        lay = QVBoxLayout(self)
        head = status.summary(commits=0).splitlines()
        self.headline = QLabel("\n".join(line for line in head if not line.startswith("  ")
                                         and not line.startswith("What's new")
                                         and line not in (status.blocked_reason, status.busy_reason)))
        self.headline.setWordWrap(True)
        self.headline.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.headline)
        if status.state not in ("error", "not-a-checkout"):
            follows = QLabel(update.FOLLOWS)
            follows.setWordWrap(True)
            follows.setStyleSheet("color: palette(placeholder-text);")
            lay.addWidget(follows)
        self.news = None
        if status.state in ("behind", "diverged"):
            self.news = QTextBrowser(self)
            self.news.setOpenExternalLinks(True)
            self.news.setMarkdown(self.notes_markdown(status))
            self.news.setMinimumSize(640, 320)
            lay.addWidget(self.news, 1)
        reason = status.blocked_reason or (status.busy_reason if status.state in ("behind", "diverged") else "")
        self.reason = QLabel(reason)
        self.reason.setWordWrap(True)
        self.reason.setStyleSheet("color: #b45309;")
        self.reason.setVisible(bool(reason))
        lay.addWidget(self.reason)
        self.check = None
        buttons = QDialogButtonBox(self)
        idle_wait = status.state == "behind" and not status.blocked_reason and bool(status.busy_reason)
        if status.can_update or idle_wait:
            label = "Update when idle" if idle_wait else "Update"
            go = buttons.addButton(label, QDialogButtonBox.ButtonRole.AcceptRole)
            go.setToolTip("Pull the new commits (fast-forward only) and install what changed; then you choose "
                          "when to restart." + (" It waits until the worker on this computer is idle."
                                                if idle_wait else ""))
            go.clicked.connect(lambda: self._choose("update"))
            later = buttons.addButton("Later", QDialogButtonBox.ButtonRole.RejectRole)
            later.clicked.connect(self.reject)
            go.setDefault(True)
            self.check = QCheckBox(UPGRADE_TEXT, self)
            self.check.setToolTip(UPGRADE_TIP)
            lay.addWidget(self.check)
        elif status.state in ("up-to-date", "ahead"):
            up = buttons.addButton("Upgrade packages...", QDialogButtonBox.ButtonRole.AcceptRole)
            up.setToolTip(UPGRADE_TIP)
            up.clicked.connect(lambda: self._choose("upgrade"))
            close = buttons.addButton(QDialogButtonBox.StandardButton.Close)
            close.clicked.connect(self.reject)
            close.setDefault(True)
        else:
            close = buttons.addButton(QDialogButtonBox.StandardButton.Close)
            close.clicked.connect(self.reject)
        lay.addWidget(buttons)
        self.buttons = buttons

    @staticmethod
    def notes_markdown(status) -> str:
        parts = []
        if status.news:
            parts += ["## What's new", status.news_markdown()]
        if status.incoming:
            parts.append(f"## Commits ({status.behind})")
            parts.append("\n".join(f"- `{c.split(' ', 1)[0]}` {c.split(' ', 1)[1] if ' ' in c else ''}"
                                   for c in status.incoming[:200]))
            if len(status.incoming) > 200:
                parts.append(f"*... and {len(status.incoming) - 200} more*")
        return "\n\n".join(parts) or "No CHANGELOG entries came with these commits."

    def _choose(self, what: str) -> None:
        self.choice = "upgrade" if what == "upgrade" or (self.check is not None and self.check.isChecked()) \
            else "update"
        self.accept()

    def button_texts(self) -> list[str]:
        return [b.text().replace("&", "") for b in self.buttons.buttons()]

    def click(self, text: str) -> None:
        """Press the button with this text (scripted tests and smoke runs)."""
        next(b for b in self.buttons.buttons() if b.text().replace("&", "") == text).click()


def _ask_update(win, status):
    """The dialog. Returns False (Later, Close), "update" (Update) or "upgrade" (the same with "Also upgrade the
    installed packages" ticked, or "Upgrade packages..." when there is nothing to pull)."""
    dlg = UpdateDialog(status, win)
    dlg.exec()
    return dlg.choice


# ------------------------------------------------------------------------------------- waiting for work
def window_busy(win) -> str:
    """What runs in this window that an update must not change under ('' when nothing does)."""
    parts = []
    if getattr(win, "worker", None) is not None:
        parts.append("a build is running in this window")
    queue = getattr(win, "job_queue", None)
    if queue is not None:
        try:
            waiting = len(queue.queued())
        except Exception:
            waiting = 0
        if waiting:
            parts.append(f"{waiting} job{'s are' if waiting != 1 else ' is'} queued here")
    return " and ".join(parts)


class _Wait(QObject):
    """An update that waits for this window's work and the worker on this computer to be idle."""

    def __init__(self, win, repo, upgrade: bool, worker_busy: str):
        super().__init__(win)
        self.win, self.repo, self.upgrade = win, repo, upgrade
        self.worker_busy = worker_busy
        self._ticks = 0
        self._probing = False
        self.timer = QTimer(self)
        self.timer.setInterval(WAIT_MS)
        self.timer.timeout.connect(self.tick)
        win._update_wait = self
        self.timer.start()
        self._show()

    def why(self) -> str:
        parts = [p for p in (window_busy(self.win), "the worker on this computer has jobs" if self.worker_busy
                             else "") if p]
        return ("waiting until " + " and ".join(parts) + " finish") if parts else "about to start"

    def _show(self) -> None:
        self.win.statusBar().showMessage(f"Update {self.why()} (Help > Check for updates... cancels it).")

    def cancel(self) -> None:
        self.timer.stop()
        if getattr(self.win, "_update_wait", None) is self:
            self.win._update_wait = None
        self.win.statusBar().showMessage("The waiting update was cancelled.", 5000)
        self.deleteLater()

    def tick(self) -> None:
        if self._probing:
            return
        self._ticks += 1
        if window_busy(self.win):
            self._show()
            return
        if self.worker_busy and self._ticks % WORKER_EVERY and self._ticks > 1:
            self._show()
            return
        self._probing = True
        from .. import update
        from . import workers

        def probed(worker):
            self._probing = False
            if getattr(self.win, "_update_wait", None) is not self:
                return                                       # cancelled meanwhile
            self.worker_busy = worker.busy_reason() if worker is not None else ""
            if self.worker_busy or window_busy(self.win):
                self._show()
                return
            self.timer.stop()
            self.win._update_wait = None
            self.deleteLater()
            run_update(self.win, self.repo, upgrade_packages=self.upgrade)

        def failed(_msg):
            self._probing = False

        root = update.find_checkout(self.repo)
        workers.start(workers.FunctionWorker(update.probe_worker, root), self.win, on_done=probed,
                      on_failed=failed)


def _ask_wait(win, what: str) -> bool:
    box = QMessageBox(win)
    box.setWindowTitle(TITLE)
    box.setIcon(QMessageBox.Icon.Information)
    box.setText(f"Not now: {what}.\n\nAn update changes the code and packages under running work, so it waits "
                "until that work is done. Jobs on other machines go on there and do not hold it.")
    wait = box.addButton("Update when they finish", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(wait)
    box.exec()
    return box.clickedButton() is wait


def show_status(win, status, repo=None) -> None:
    choice = _ask_update(win, status)
    if not choice:
        return
    upgrade = choice == "upgrade"
    here = window_busy(win)
    worker_busy = status.busy_reason
    if here or worker_busy:
        what = " and ".join(p for p in (here, "the worker on this computer runs jobs from this checkout"
                                        if worker_busy else "") if p)
        if _ask_wait(win, what):
            _Wait(win, repo, upgrade, worker_busy)
        return
    run_update(win, repo, upgrade_packages=upgrade)


# --------------------------------------------------------------------------------------------- the result
def _show_result(win, res) -> tuple[bool, bool]:
    """What the update did; returns (restart the app now, restart this computer's worker)."""
    box = QMessageBox(win)
    box.setWindowTitle(TITLE)
    box.setIcon(QMessageBox.Icon.Information if res.ok else QMessageBox.Icon.Warning)
    lines, shown = [], 0
    for line in res.message.splitlines():
        if line.startswith("  ") and line.strip() in res.package_changes:
            shown += 1
            if shown > SHOWN_CHANGES:
                continue
        lines.append(line)
    if shown > SHOWN_CHANGES:
        lines.append(f"  ... and {shown - SHOWN_CHANGES} more (Show Details)")
    box.setText("\n".join(lines))
    details = "\n".join(res.package_changes + ([""] if res.package_changes else []) + [res.log]).strip()
    if details:
        box.setDetailedText(details)
    worker = res.worker if res.worker_needs_restart else None
    check = None
    if worker is not None and worker.can_restart:
        how = "LaunchAgent" if worker.manager == "launchd" else "scheduled task"
        check = QCheckBox(f"Restart this computer's worker too (it is idle; {how})", box)
        check.setChecked(True)
        check.setToolTip("A running worker keeps the code it started with. This restarts it now, whichever "
                         "button you press; its queue and pause carry over (docs/WORKERS.md).")
        box.setCheckBox(check)
    restart_btn = None
    if res.restart_needed:
        restart_btn = box.addButton("Restart now", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(restart_btn)
    else:
        box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    return (restart_btn is not None and box.clickedButton() is restart_btn,
            check is not None and check.isChecked())


def run_update(win, repo=None, upgrade_packages: bool = False) -> None:
    """Pull and reinstall (and with ``upgrade_packages``, upgrade the installed packages) on a worker thread,
    then show what happened and offer to restart."""
    from .. import update
    from . import workers

    if upgrade_packages:
        _busy(win, "Updating QuantaBench and upgrading its packages with pip (this takes a few minutes)...")
    else:
        _busy(win, "Updating QuantaBench (git pull, then pip if the dependencies changed)...")

    def done(res):
        _busy(win, None)
        if hasattr(win, "log"):
            for line in res.message.splitlines():
                win.log(line)
        restart_app, restart_worker = _show_result(win, res)
        if restart_worker:
            _restart_worker(win, res.worker, then=(lambda: restart(win)) if restart_app else None)
        elif restart_app:
            restart(win)
        elif res.restart_needed:
            win.statusBar().showMessage("QuantaBench was updated: restart it to use the new version.")

    def failed(msg):
        _busy(win, None)
        QMessageBox.warning(win, TITLE, msg)

    kwargs = {"upgrade_packages": True} if upgrade_packages else {}
    workers.start(workers.FunctionWorker(update.update, repo, **kwargs), win, on_done=done, on_failed=failed)


def _restart_worker(win, worker, then=None) -> None:
    """``local_worker.restart`` on a worker thread; says what happened, then calls ``then``."""
    from .. import local_worker
    from . import workers

    win.statusBar().showMessage("Restarting this computer's worker...")

    def done(result):
        ok, msg = result
        if hasattr(win, "log"):
            win.log(msg)
        win.statusBar().showMessage(msg)
        if not ok and then is None:
            QMessageBox.warning(win, TITLE, msg)
        if then is not None:
            then()

    def failed(msg):
        if hasattr(win, "log"):
            win.log(f"Restarting the worker failed: {msg}")
        if then is not None:
            then()

    workers.start(workers.FunctionWorker(local_worker.restart, worker), win, on_done=done, on_failed=failed)


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


# ------------------------------------------------------------------------------------------ app shortcuts
def create_shortcuts(win) -> None:
    """Help > Create app shortcuts...: the launchers of ``quantabench-cli shortcuts``, after asking."""
    from .. import launchers
    from ..settings import repo_root
    from . import workers

    repo = repo_root()
    variants = launchers.find_variants(repo)
    box = QMessageBox(win)
    box.setWindowTitle(SHORTCUTS_TITLE)
    box.setIcon(QMessageBox.Icon.Question)
    names = "\n".join(f"  {v.label}  ({v.prefix.name})" for v in variants)
    box.setText(f"Make {launchers.what()} that start QuantaBench without a terminal window, with its icon?\n\n"
                f"{names}\n\nThey point at this checkout, so updates need no new ones; make them again if the "
                "checkout moves. (quantabench-cli shortcuts does the same.)")
    desktop = QCheckBox("Also on the Desktop", box)
    box.setCheckBox(desktop)
    make = box.addButton("Create", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(make)
    box.exec()
    if box.clickedButton() is not make:
        return

    def done(paths):
        QMessageBox.information(win, SHORTCUTS_TITLE, "Made:\n" + "\n".join(f"  {p}" for p in paths))

    def failed(msg):
        QMessageBox.warning(win, SHORTCUTS_TITLE, f"Could not make the shortcuts:\n{msg}")

    workers.start(workers.FunctionWorker(launchers.create, repo, desktop=desktop.isChecked(), variants=variants),
                  win, on_done=done, on_failed=failed)
