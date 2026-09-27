"""Shared E1 plumbing for every runner: flags, frozen-set wiring, and the run record.

Git state is captured at IMPORT, i.e. when the running code was read from disk, not at exit;
a pull or edit mid-run is detected and recorded rather than silently re-attributed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent


def _git(*args: str) -> str:
    """Raw stdout. Callers strip what they need: `git status --porcelain` lines start with a
    space-padded status column, so stripping the whole output would eat the first path's first
    character."""
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_state() -> Dict:
    # -z: NUL-separated, no quoting of odd paths; each entry is "XY path"
    raw = _git("status", "--porcelain=v1", "-z", "--untracked-files=no")
    dirty = [e[3:] for e in raw.split("\0") if len(e) > 3]
    return {"commit": _git("rev-parse", "HEAD").strip() or None,
            "branch": _git("branch", "--show-current").strip() or None,
            "dirty_tracked_files": dirty}


GIT_AT_IMPORT = _git_state()


def add_e1_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared by every runner. With none of them, a runner behaves exactly as before."""
    g = ap.add_argument_group("E1 (frozen evaluation set)")
    g.add_argument("--eval_set", type=str, default=None,
                   help="path to frozen_eval_v1.json; omit for the original behaviour")
    g.add_argument("--fit_pool", choices=["heldout", "full"], default="heldout",
                   help="heldout = fit without the eval questions (E1 default); "
                        "full = fit on every unique pair (leak-control arms only)")
    g.add_argument("--out_dir", type=str, default=None,
                   help="write checkpoints and results here instead of the default paths")
    g.add_argument("--seed", type=int, default=None, help="overrides the runner's RANDOM_SEED")
    g.add_argument("--dry_run", action="store_true",
                   help="build data and the fit pool, print sizes and hashes, exit before loading the model")


def check_e1_args(args: argparse.Namespace) -> None:
    if args.eval_set and not args.out_dir:
        raise SystemExit("--eval_set requires --out_dir: E1 results must never share a checkpoint "
                         "directory with earlier runs or with another fit variant.")
    if args.fit_pool == "full" and not args.eval_set:
        raise SystemExit("--fit_pool full only makes sense with --eval_set.")


def frozen_split(filtered_pairs: List[Dict], args: argparse.Namespace):
    """Returns (fz, fit_pairs, eval_pairs) for an E1 run, or (None, None, None) otherwise."""
    if not args.eval_set:
        return None, None, None
    import frozen_eval as fe
    fz = fe.load(Path(args.eval_set))
    fit = fe.fit_pool(filtered_pairs, fz, full=(args.fit_pool == "full"))
    return fz, fit, fe.eval_pairs(fz)


def e1_metadata(args: argparse.Namespace, fz: Optional[Dict], fit_size: Optional[int]) -> Dict:
    return {
        "frozen_eval_sha256": fz["sha256"] if fz else None,
        "fit_pool": args.fit_pool if fz else None,
        "fit_pool_size": fit_size,
        "seed": args.seed,
        "git_at_import": GIT_AT_IMPORT,
        "argv_flags": {k: v for k, v in vars(args).items()},
    }


class RunRecord:
    """Writes out_dir/run_record.json on entry (status=running) and on exit (ok/failed).

    Written on entry too, so a run killed by the pod dying still leaves a record saying so.
    """

    def __init__(self, out_dir: Path, runner: str, args: argparse.Namespace, fz: Optional[Dict],
                 name: str = "run_record"):
        # arms that share a directory (e.g. the rank sweep reusing one SVD) need distinct names
        self.path = Path(out_dir) / f"{name}.json"
        self.rec = {"runner": runner, "status": "running",
                    "started": _dt.datetime.now().isoformat(timespec="seconds"),
                    "host": platform.node(),
                    "frozen_eval_sha256": fz["sha256"] if fz else None,
                    "fit_pool": args.fit_pool if fz else None,
                    "seed": args.seed, "flags": {k: v for k, v in vars(args).items()},
                    "git_at_import": GIT_AT_IMPORT, "results": {}}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rec, indent=1, default=str))
        tmp.replace(self.path)

    def __enter__(self) -> "RunRecord":
        self._write()
        return self

    def result(self, **kw) -> None:
        self.rec["results"].update(kw)
        self._write()

    def __exit__(self, exc_type, exc, tb) -> bool:
        now = _git_state()
        self.rec["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        self.rec["status"] = "ok" if exc_type is None else f"failed: {exc_type.__name__}: {exc}"
        self.rec["git_changed_during_run"] = (now["commit"] != GIT_AT_IMPORT["commit"]
                                              or now["dirty_tracked_files"] != GIT_AT_IMPORT["dirty_tracked_files"])
        self._write()
        return False
