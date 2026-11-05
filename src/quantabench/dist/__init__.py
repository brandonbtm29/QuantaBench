"""Splitting work across processing units and machines ("QuantaBench grid").

Three layers, each usable on its own (plan: docs/PLATFORMS.md, sections 2-3):

``dist.lanes``    one machine, several **lanes**: worker processes that each own a device and a thread
                  budget (e.g. 1 GPU lane + 5 CPU lanes of 2 threads on BTM NOVA). ``jobs.run_resumable``
                  pulls molecules into free lanes, so a fast lane simply takes more of them.
``dist.profile``  what a machine can do: hardware, engines, external programs and measured throughput.
                  Saved as ``machines/<name>.json`` in the (synced) settings folder, so every machine
                  sees the others.
``dist.plan``     deterministic, weighted splitting of a molecule list between machines (or lanes).
``dist.worker``   ``quantabench-cli worker``: an HTTP/JSON service on the Tailscale address that runs jobs.
``dist.client``   submitting to workers and ``quantabench-cli grid build``: split by throughput, submit,
                  follow progress, merge the result tables.

Rules that hold everywhere: one writer per run folder (a worker writes its own run folder under the
shared Runs folder); molecule ids decide chunk membership (SHA-256, stable across machines and list
order); a worker refuses jobs from a different QuantaBench version.
"""
