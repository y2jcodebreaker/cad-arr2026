"""Loader for the frozen E1 evaluation set. Every runner imports this.

    import frozen_eval as fe
    fz = fe.load()                                   # verifies sha256, refuses a changed file
    prompts_src = fe.eval_pairs(fz)                  # 250 items, frozen order
    fit = fe.fit_pool(filtered_pairs, fz)            # held-out fit pool, asserted size
    fit = fe.fit_pool(filtered_pairs, fz, full=True) # leak-control arms only

`filtered_pairs` is whatever the runner's own loudness filter produced (2,083 entries, with
duplicated rows). fit_pool() collapses the duplicates and keeps only the pairs the frozen set
assigns to fitting, then ASSERTS the size. A runner whose pool drifted (for instance because it
recomputed the loudness norms) therefore stops here instead of fitting on something else.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

FROZEN = Path(__file__).resolve().parent / "frozen_eval_v1.json"
EXPECTED_SHA256 = "153439a63463081fe0773c26c6a259f1bc6a9cd4eafd40ffa49fab9aaf807658"


def pair_key(clean: str, corrupted: str) -> str:
    """Identity of a contrastive pair: its text. Shared by the builder and every runner."""
    return hashlib.sha256(f"{clean}\x00{corrupted}".encode()).hexdigest()


def content_sha256(fz: Dict) -> str:
    body = {k: v for k, v in fz.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(path: Path = FROZEN) -> Dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing. It is committed in the repo; run `bash pod_pull.sh`.")
    fz = json.loads(Path(path).read_text())
    actual = content_sha256(fz)
    if actual != fz.get("sha256"):
        raise ValueError(f"{Path(path).name} was edited: content hash {actual[:16]} != recorded "
                         f"{str(fz.get('sha256'))[:16]}. Restore it with git.")
    if actual != EXPECTED_SHA256:
        raise ValueError(f"{Path(path).name} is not the pre-registered set: {actual[:16]} != "
                         f"{EXPECTED_SHA256[:16]}.")
    assert len(fz["eval"]) == fz["n_eval"] == 250, "eval set size changed"
    return fz


def eval_pairs(fz: Dict) -> List[Dict]:
    """The 250 evaluation items in frozen order, in the {clean_text, corrupted_text} shape the
    runners already use, plus metadata the analysis needs."""
    return [{"clean_text": e["clean_text"], "corrupted_text": e["corrupted_text"],
             "question_id": e["question_id"], "category": e["category"], "domain": e["domain"]}
            for e in fz["eval"]]


def fit_pool(filtered_pairs: List[Dict], fz: Dict, full: bool = False) -> List[Dict]:
    """Deduplicate `filtered_pairs` and keep the pairs this arm may be fit on.

    full=False: held-out pool (the eval questions removed).        Expected 633.
    full=True : every unique pair, eval questions included. Only for the leak-control arms.
    """
    allowed = None if full else set(fz["fit_pair_keys"])
    seen, out = set(), []
    for p in filtered_pairs:
        k = pair_key(p["clean_text"], p["corrupted_text"])
        if k in seen or (allowed is not None and k not in allowed):
            continue
        seen.add(k)
        out.append(p)
    want = fz["expected_full_pool_size"] if full else fz["expected_fit_pool_size"]
    if len(out) != want:
        raise RuntimeError(
            f"fit pool has {len(out)} unique pairs, the frozen set expects {want} "
            f"({'full' if full else 'held-out'}). This runner's loudness-filtered pool differs "
            f"from the one the set was built on; check angular_checkpoints/accesseval_loudness.npz.")
    eval_keys = {pair_key(e["clean_text"], e["corrupted_text"]) for e in fz["eval"]}
    if not full:
        assert not eval_keys & seen, "LEAK: an eval item reached the held-out fit pool"
    return out


def describe(fz: Dict, full: bool = False) -> str:
    size = fz["expected_full_pool_size"] if full else fz["expected_fit_pool_size"]
    return (f"{fz['version']} sha256 {fz['sha256'][:16]} | eval {fz['n_eval']} items / "
            f"{len(fz['eval_question_ids'])} questions | fit {size} unique pairs "
            f"({'FULL, leak control' if full else 'held-out'})")
