"""Score E1 (frozen evaluation set) against the pre-registration, claims/PREREG_E1_frozen_eval.md.

CPU only. Reads results/e1/ (committed evidence), writes analysis-output/e1/e1_scores.json and
prints every prediction's verdict. Criteria are the registered ones; nothing here is tuned.

Conventions (fixed before looking at any judge score):
- Reference = unsteered seed 42 from the CAD runner (`cad_heldout/baseline_seed42`). Every arm
  shares its prompts, template, temperature and max_new_tokens. d = cohens_d(reference, arm), the
  runner's own function; positive d = less medicalized.
- Medicalization is RECOMPUTED from text with the pinned scorer for every arm, and checked against
  the score each runner stored.
- Quality = mean judge score; parse failures stay at 5.0, as registered. Quality delta is paired on
  the 250 items (arm minus reference).
- Intervals: question-level bootstrap, 50 questions (5 items each), 2,000 resamples, seed 0.
- Degeneracy (text only, frontier thresholds): unique-text ratio < 0.50, Llama parse-failure rate
  > 0.10, or mean medicalization < -0.05.
- Per D1, the registered "k=40" arm is cad_rank_k40; k=auto (33/36) is reported beside it.

    python e1_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
import evaluate_rtsd_fullres_generation as rt  # noqa: E402  (pinned scorer, cohens_d)
import frozen_eval as fe  # noqa: E402
import judge_sweep_outputs as js  # noqa: E402  (text_key: the judge's own hash)

E1 = Path("results/e1")
OUT = Path("analysis-output/e1")
N_BOOT, BOOT_SEED = 2000, 0
QUALITY_WINDOW = 1.0   # P3/P6: "within 1.0 of the unsteered mean"

CATEGORY_STEMS = {"Genetic": "genetic", "Hearing": "hearing", "Learning": "learning",
                  "Mental": "mental", "Mobility": "mobility", "Neurological": "neurolog",
                  "Sensory": "sensory", "Speech": "speech", "Vision": "vision"}


def _texts(obj) -> list[str]:
    return [r["response"] if isinstance(r, dict) else r for r in obj]


def _stored(obj) -> list[float] | None:
    if obj and isinstance(obj[0], dict) and "medicalization_score" in obj[0]:
        return [float(r["medicalization_score"]) for r in obj]
    return None


def load_arms() -> dict[str, dict]:
    j = lambda p: json.loads((E1 / p).read_text())  # noqa: E731
    arms: dict[str, list] = {}
    for s in (42, 43, 44):
        arms[f"unsteered_s{s}"] = j(f"cad_heldout/baseline_seed{s}_responses.json")["responses"]
    arms["unsteered_leace_runner"] = j("leace/base_seed42.json")["responses"]
    arms["cad_proj_kauto"] = j("cad_heldout/proj_alpha1.0_responses.json")["responses"]
    for k in (1, 5, 10, 20, 40, 80):
        arms[f"cad_proj_k{k}"] = j(f"cad_heldout/proj_k{k}_alpha1.0_responses.json")["responses"]
    arms["cad_proj_probe"] = j("cad_heldout/proj_probebasis_alpha1.0_responses.json")["responses"]
    arms["cad_proj_meandiff"] = j("cad_heldout/proj_meandiffbasis_alpha1.0_responses.json")["responses"]
    for a in (3, 5, 7, 9):
        arms[f"add_probe_a{a}"] = j(f"cad_heldout/probe_alpha{a}.0_responses.json")["responses"]
    for a in (3, 5, 7, 8):
        arms[f"add_meandiff_a{a}"] = j(f"cad_heldout/meandiff_alpha{a}.0_responses.json")["responses"]
    arms["cad_proj_kauto_FULLPOOL"] = j("cad_full/proj_alpha1.0_responses.json")["responses"]
    for k in (1, 5, 10, 20, 40, 80):
        arms[f"leace_k{k}"] = j(f"leace/dual_k{k}.json")["responses"]
    arms["caa"] = j("caa_heldout/caa_results.json")["accesseval"]["layers"]["L14"]["alpha_1.0"]["responses"]
    arms["caa_FULLPOOL"] = j("caa_full/caa_results.json")["accesseval"]["layers"]["L14"]["alpha_1.0"]["responses"]
    arms["fairsteer"] = j("fairsteer/fairsteer_results.json")["accesseval"]["layers"]["L29"]["thresh_0.5"]["alpha_2.0"]["responses"]
    arms["sadi"] = j("sadi/sadi_results.json")["accesseval"]["strengths"]["strength_10.0"]["responses"]
    arms["angular"] = j("angular/angular_results.json")["accesseval"]["strategies"]["max_sim_L23"]["mode_0"]["angle_150"]["responses"]
    # A7: each baseline at its authors' own setting, scored once they exist
    for name, f, keys in (("angular_adaptive", "angular_adaptive/angular_results.json",
                           ("accesseval", "strategies", "max_sim_L23", "mode_1", "angle_150", "responses")),
                          ("caa_m2", "caa_m2/caa_results.json", ("accesseval", "layers", "L14", "alpha_2.0", "responses")),
                          ("fairsteer_a1", "fairsteer_a1/fairsteer_results.json",
                           ("accesseval", "layers", "L29", "thresh_0.5", "alpha_1.0", "responses")),
                          ("sadi_s5", "sadi_s5/sadi_results.json", ("accesseval", "strengths", "strength_5.0", "responses"))):
        if (E1 / f).exists():
            o = j(f)
            for k in keys:
                o = o[k]
            arms[name] = o
    # A4 prompting arms, scored once they exist
    for name, f in (("prompt_min", "cad_heldout/baseline_seed42_sysmin_responses.json"),
                    ("prompt_explicit", "cad_heldout/baseline_seed42_sysexplicit_responses.json"),
                    ("prompt_explicit_cad_k40", "cad_heldout/proj_k40_sysexplicit_alpha1.0_responses.json")):
        if (E1 / f).exists():
            arms[name] = j(f)["responses"]
    out = {}
    for name, raw in arms.items():
        assert len(raw) == 250, f"{name}: {len(raw)} responses"
        out[name] = {"texts": _texts(raw), "stored_med": _stored(raw)}
    return out


def check_alignment(arms: dict, ev: list[dict]) -> dict[str, float]:
    """Response i must answer eval item i. CAA stores its prompt: exact check. For the rest, the
    fraction of responses naming item i's disability category, against the same with the order
    rotated by one item (chance level)."""
    caa = json.loads((E1 / "caa_heldout/caa_results.json").read_text())
    stored = caa["accesseval"]["layers"]["L14"]["alpha_1.0"]["responses"]
    assert [r["corrupted_text"] for r in stored] == [e["corrupted_text"] for e in ev], "CAA order"
    stems = [CATEGORY_STEMS[e["category"].split()[0]] for e in ev]
    res = {}
    for name, a in arms.items():
        t = [x.lower() for x in a["texts"]]
        hit = np.mean([s in x for s, x in zip(stems, t)])
        rot = np.mean([s in x for s, x in zip(stems[1:] + stems[:1], t)])
        res[name] = (float(hit), float(rot))
    return res


def judge_map(name: str) -> tuple[dict, dict]:
    d = json.loads((E1 / f"judge_{name}.json").read_text())
    return d["scores_by_hash"], d.get("parsed_by_hash", {})


def boot_indices(qids: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(BOOT_SEED)
    uq = np.unique(qids)
    by_q = {q: np.flatnonzero(qids == q) for q in uq}
    return [np.concatenate([by_q[q] for q in rng.choice(uq, len(uq), replace=True)])
            for _ in range(N_BOOT)]


def ci(x) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main() -> int:
    fz = fe.load()
    ev = fe.eval_pairs(fz)
    qids = np.array([e["question_id"] for e in ev])
    arms = load_arms()

    align = check_alignment(arms, ev)
    worst = min(h - r for h, r in align.values())
    print(f"alignment: category named in response, own item vs rotated; worst margin {worst:+.2f}")
    for n, (h, r) in align.items():
        if h - r < 0.10:
            print(f"   WEAK ALIGNMENT {n}: {h:.2f} vs {r:.2f}")

    llama, llama_ok = judge_map("llama")
    qwen, _ = judge_map("second")
    for name, a in arms.items():
        a["med"] = np.array([rt.medicalization_score(t) for t in a["texts"]])
        if a["stored_med"] is not None:
            diff = np.max(np.abs(a["med"] - np.array(a["stored_med"])))
            assert diff < 1e-9, f"{name}: recomputed medicalization differs from stored by {diff}"
        keys = [js.text_key(t) for t in a["texts"]]
        a["q_llama"] = np.array([llama[k] for k in keys], dtype=float)
        a["q_qwen"] = np.array([qwen[k] for k in keys], dtype=float)
        a["unique"] = len(set(a["texts"])) / 250
        a["parse_fail"] = float(np.mean([not llama_ok.get(k, True) for k in keys]))

    ref = arms["unsteered_s42"]
    boots = boot_indices(qids)

    def d_of(a, idx=None):
        idx = slice(None) if idx is None else idx
        return rt.cohens_d(ref["med"][idx], a["med"][idx])

    summary = {}
    for name, a in arms.items():
        dq_l = a["q_llama"] - ref["q_llama"]
        dq_q = a["q_qwen"] - ref["q_qwen"]
        bd = [d_of(a, i) for i in boots]
        flags = [f for f, bad in (("repetition", a["unique"] < 0.50),
                                  ("unparseable", a["parse_fail"] > 0.10),
                                  ("overshoot", a["med"].mean() < -0.05)) if bad]
        summary[name] = dict(
            med=float(a["med"].mean()), d=d_of(a), d_ci=ci(bd),
            dmed_pct=float(100 * (ref["med"].mean() - a["med"].mean()) / ref["med"].mean()),
            q_llama=float(a["q_llama"].mean()), q_qwen=float(a["q_qwen"].mean()),
            dq_llama=float(dq_l.mean()), dq_llama_ci=ci([dq_l[i].mean() for i in boots]),
            dq_qwen=float(dq_q.mean()), dq_qwen_ci=ci([dq_q[i].mean() for i in boots]),
            unique=a["unique"], parse_fail=a["parse_fail"], degenerate=flags,
            alignment=align[name])
        a["_bd"] = np.array(bd)

    print(f"\n{'arm':<26}{'med':>7}{'d':>7}{'  d 95% CI':>16}{'ΔMed%':>7}"
          f"{'ΔQ llama':>10}{'ΔQ qwen':>9}  flags")
    for n, s in summary.items():
        print(f"{n:<26}{s['med']:>7.3f}{s['d']:>7.3f}  [{s['d_ci'][0]:>5.2f},{s['d_ci'][1]:>5.2f}]"
              f"{s['dmed_pct']:>7.1f}{s['dq_llama']:>+10.2f}{s['dq_qwen']:>+9.2f}  {','.join(s['degenerate'])}")

    verdicts = {}
    S = summary

    # ---- P1: SADI largest quality drop, CAD projection smallest, among the five published points
    pub = {"CAD projection (k=40)": "cad_proj_k40", "CAA": "caa", "FairSteer": "fairsteer",
           "SADI": "sadi", "Angular": "angular"}
    p1 = {}
    for j in ("llama", "qwen"):
        drops = {lbl: -S[a][f"dq_{j}"] for lbl, a in pub.items()}
        order = sorted(drops, key=drops.get, reverse=True)
        p1[j] = dict(order_largest_drop_first=order, drops=drops,
                     sadi_largest=order[0] == "SADI", cad_smallest=order[-1] == "CAD projection (k=40)")
    verdicts["P1"] = dict(detail=p1, supported=all(v["sadi_largest"] and v["cad_smallest"] for v in p1.values()))

    # ---- P2: paired CAD-projection minus CAA quality, I3 (Qwen) CI excludes 0; I2 reported
    p2 = {}
    for j in ("qwen", "llama"):
        diff = arms["cad_proj_k40"][f"q_{j}"] - arms["caa"][f"q_{j}"]
        p2[j] = dict(mean=float(diff.mean()), ci=ci([diff[i].mean() for i in boots]))
    p2["supported"] = p2["qwen"]["ci"][0] > 0 or p2["qwen"]["ci"][1] < 0
    verdicts["P2"] = p2

    # ---- P3: for both operators, d(k=40) > d(k=1) with non-overlapping CIs, and k=40 quality within 1.0
    p3 = {}
    for op, k1, k40 in (("projection", "cad_proj_k1", "cad_proj_k40"), ("LEACE", "leace_k1", "leace_k40")):
        a1, a40 = S[k1], S[k40]
        sep = a40["d_ci"][0] > a1["d_ci"][1]
        qok = abs(a40["dq_llama"]) <= QUALITY_WINDOW and abs(a40["dq_qwen"]) <= QUALITY_WINDOW
        p3[op] = dict(d_k1=a1["d"], d_k1_ci=a1["d_ci"], d_k40=a40["d"], d_k40_ci=a40["d_ci"],
                      cis_separate=sep, k40_quality_within_1=qok,
                      k1_within_005_of_k40=abs(a40["d"] - a1["d"]) <= 0.05)
    verdicts["P3"] = dict(detail=p3, supported=all(v["cis_separate"] and v["k40_quality_within_1"] for v in p3.values()))

    # ---- P5: full-pool fit raises d by < 0.10 for both CAD projection and CAA
    p5 = {}
    for m, h, f in (("CAD projection (k=auto)", "cad_proj_kauto", "cad_proj_kauto_FULLPOOL"), ("CAA", "caa", "caa_FULLPOOL")):
        rise = S[f]["d"] - S[h]["d"]
        p5[m] = dict(heldout=S[h]["d"], full=S[f]["d"], rise=rise,
                     rise_ci=ci(arms[f]["_bd"] - arms[h]["_bd"]))
    rises = [v["rise"] for v in p5.values()]
    verdicts["P5"] = dict(detail=p5, supported=all(r < 0.10 for r in rises),
                          outcome="negligible (<0.05 both)" if all(r < 0.05 for r in rises) else
                                  "F5 fired" if any(r >= 0.10 for r in rises) else "between 0.05 and 0.10")

    # ---- P6: direction vs rank, on the matched alpha=1 projections
    kp, k1, k40 = S["cad_proj_probe"], S["cad_proj_k1"], S["cad_proj_k40"]
    working = {n: abs(S[n]["dq_llama"]) <= QUALITY_WINDOW and abs(S[n]["dq_qwen"]) <= QUALITY_WINDOW
               for n in ("cad_proj_probe", "cad_proj_k1", "cad_proj_k40")}
    a_ = abs(kp["d"] - k40["d"]) <= 0.10
    b_ = (kp["d"] - k1["d"] <= 0.05) and k40["d_ci"][0] > max(kp["d_ci"][1], k1["d_ci"][1])
    verdicts["P6"] = dict(d_probe_proj=kp["d"], d_probe_ci=kp["d_ci"], d_svd_k1=k1["d"], d_svd_k40=k40["d"],
                          d_svd_k40_ci=k40["d_ci"], working=working,
                          outcome="(a) direction, not rank" if a_ else "(b) rank, not direction" if b_ else "(c) both")

    # ---- P7: removal vs addition at matched d, per direction, both judges
    p7 = {}
    for direction, rem, adds in (("probe", "cad_proj_probe", [f"add_probe_a{a}" for a in (3, 5, 7, 9)]),
                                 ("meandiff", "cad_proj_meandiff", [f"add_meandiff_a{a}" for a in (3, 5, 7, 8)])):
        d_rem = S[rem]["d"]
        ds = [S[a]["d"] for a in adds]
        pair = next(((adds[i], adds[i + 1]) for i in range(len(adds) - 1)
                     if min(ds[i], ds[i + 1]) <= d_rem <= max(ds[i], ds[i + 1])), None)
        if pair is None:
            p7[direction] = dict(d_rem=d_rem, additive_d=dict(zip(adds, ds)), matched=False)
            continue
        lo, hi = pair
        res = dict(d_rem=d_rem, bracket=[lo, hi], additive_d=dict(zip(adds, ds)), matched=True)
        for j in ("llama", "qwen"):
            def delta(idx=None):
                dr, dl, dh = (d_of(arms[x], idx) for x in (rem, lo, hi))
                sl = slice(None) if idx is None else idx
                ql, qh, qr = (arms[x][f"q_{j}"][sl].mean() for x in (lo, hi, rem))
                w = 0.0 if dh == dl else (dr - dl) / (dh - dl)
                return qr - (ql + w * (qh - ql)), not (min(dl, dh) <= dr <= max(dl, dh))
            pt, _ = delta()
            bs = [delta(i) for i in boots]
            res[j] = dict(delta=float(pt), ci=ci([b[0] for b in bs]),
                          frac_resamples_outside_bracket=float(np.mean([b[1] for b in bs])))
        s_l, s_q = res["llama"], res["qwen"]
        excl = lambda c: c[0] > 0 or c[1] < 0  # noqa: E731
        res["supported_removal_better"] = (s_l["delta"] > 0 and s_q["delta"] > 0
                                           and excl(s_l["ci"]) and excl(s_q["ci"]))
        res["supported_addition_better"] = (s_l["delta"] < 0 and s_q["delta"] < 0
                                            and excl(s_l["ci"]) and excl(s_q["ci"]))
        p7[direction] = res
    n_rem = sum(bool(v.get("supported_removal_better")) for v in p7.values())
    if not any(v["matched"] for v in p7.values()):
        # registered rule: outside the additive range is "unmatched"; report it, never extrapolate
        outcome = "unmatched in both directions: P7 not testable, so the principle is not claimed"
    else:
        outcome = {2: "(a) removal preserves quality in both directions",
                   1: "(b) one direction only"}.get(n_rem, "(c) neither")
    verdicts["P7"] = dict(detail=p7, outcome=outcome)

    # ---- P8 (A4): prompting baseline
    if "prompt_explicit" in arms:
        pe, ck = arms["prompt_explicit"], arms["cad_proj_k40"]
        dd = S["cad_proj_k40"]["d"] - S["prompt_explicit"]["d"]
        dd_ci = ci(ck["_bd"] - pe["_bd"])
        qd = {}
        for j in ("llama", "qwen"):
            diff = pe[f"q_{j}"] - ck[f"q_{j}"]
            qd[j] = dict(mean=float(diff.mean()), ci=ci([diff[i].mean() for i in boots]))
        clean = not S["prompt_explicit"]["degenerate"]
        a_ = clean and dd <= 0.05 and all(v["mean"] >= 0 for v in qd.values())
        b_ = dd >= 0.05 and dd_ci[0] > 0
        p8 = dict(delta_d_cad_minus_prompt=dd, delta_d_ci=dd_ci, quality_prompt_minus_cad=qd,
                  prompt_flags=S["prompt_explicit"]["degenerate"],
                  prompt_min=dict(d=S["prompt_min"]["d"], dq_llama=S["prompt_min"]["dq_llama"],
                                  dq_qwen=S["prompt_min"]["dq_qwen"]) if "prompt_min" in S else None)
        outcomes = ["(a) prompting at least as good"] if a_ else []
        outcomes += ["(b) steering adds debiasing beyond prompting"] if b_ else []
        if "prompt_explicit_cad_k40" in arms:
            co = arms["prompt_explicit_cad_k40"]
            best = "prompt_explicit" if S["prompt_explicit"]["d"] >= S["cad_proj_k40"]["d"] else "cad_proj_k40"
            gain_ci = ci(co["_bd"] - arms[best]["_bd"])
            qok = all(abs(S["prompt_explicit_cad_k40"][f"dq_{j}"]) <= QUALITY_WINDOW for j in ("llama", "qwen"))
            c_ = S["prompt_explicit_cad_k40"]["d"] > S[best]["d"] and gain_ci[0] > 0 and qok
            p8["combo"] = dict(d=S["prompt_explicit_cad_k40"]["d"], gain_over=best, gain_ci=gain_ci,
                               quality_within_1=qok, complementary=c_)
            outcomes += ["(c) complementary"] if c_ else []
        p8["outcome"] = outcomes or ["(d) inconclusive"]
        verdicts["P8"] = p8

    # ---- baseline replicates: how much does d move with the reference draw alone?
    alt = {}
    for other in ("unsteered_s43", "unsteered_s44", "unsteered_leace_runner"):
        alt[other] = dict(d_vs_s42=d_of(arms[other]),
                          dq_llama=float((arms[other]["q_llama"] - ref["q_llama"]).mean()),
                          dq_qwen=float((arms[other]["q_qwen"] - ref["q_qwen"]).mean()))
    verdicts["reference_noise"] = alt

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e1_scores.json").write_text(json.dumps(
        {"conventions": __doc__, "frozen_eval_sha256": fz["sha256"], "n_boot": N_BOOT,
         "boot_seed": BOOT_SEED, "summary": summary, "verdicts": verdicts}, indent=1, default=float))
    print(f"\nsaved {OUT / 'e1_scores.json'}\n")
    print(json.dumps(verdicts, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
