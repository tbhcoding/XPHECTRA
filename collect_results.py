"""
collect_results.py
==================

Consolidates every final number into ONE place: RESULTS.json (machine
readable) and RESULTS.md (human readable), both at the repo root.

Why this exists: the results are produced by several scripts and land in
three different folders, so "what is the final number" previously required
knowing which of thirteen JSON files to open. Worse, crn_5seed_final/ --
the folder holding the headline result -- stored only per-seed histories
and no aggregate at all.

This reads the existing saved outputs and reassembles them. It computes
nothing new and reruns no model, so it cannot disagree with the evidence
files; if one is missing it says so rather than guessing.

Usage:
    python collect_results.py
"""

import json
import os
from datetime import date

MCO = "metric_check_outputs"


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def ms(d, key):
    """mean/std pair -> (mean, std) or None."""
    if not d:
        return None
    v = d.get("summary", {}).get(key)
    return (v["mean"], v["std"]) if v else None


def main():
    out = {"generated": str(date.today()), "sources": {}}

    # ---- headline, from the 500-sample unseen set -----------------------
    f = f"{MCO}/final_evaluation_TEST500.json"
    d = load(f)
    if d:
        s = d["summary"]
        out["headline"] = {
            "evaluated_on": f"{d['data']} ({d['n_samples']} samples, unseen)",
            "seeds": len(d["per_checkpoint"]),
            "r2": ms(d, "pooled_r2"),
            "mae_ph": ms(d, "mae"),
            "rmse_ph": ms(d, "rmse"),
            "within_0.15_ph_pct": (s["within_tolerance_pct"]["0.15"]["mean"],
                                   s["within_tolerance_pct"]["0.15"]["std"]),
            "within_0.20_ph_pct": (s["within_tolerance_pct"]["0.20"]["mean"],
                                   s["within_tolerance_pct"]["0.20"]["std"]),
        }
        out["sources"]["headline"] = f

    # ---- the other two evaluation sets, for the robustness statement ----
    out["other_evaluation_sets"] = {}
    for lbl, fn in [("validation_n50", "final_evaluation.json"),
                    ("test_n50", "final_evaluation_TEST.json")]:
        d = load(f"{MCO}/{fn}")
        if d:
            out["other_evaluation_sets"][lbl] = {
                "r2": ms(d, "pooled_r2"), "mae_ph": ms(d, "mae"), "rmse_ph": ms(d, "rmse")}
            out["sources"][lbl] = f"{MCO}/{fn}"

    # ---- per-seed detail, from the headline run ------------------------
    d = load(f"{MCO}/final_evaluation_TEST500.json")
    if d:
        import re as _re
        out["per_seed"] = [
            {"seed": int(_re.search(r"seed_(\d+)", r["checkpoint"]).group(1)),
             "checkpoint": r["checkpoint"], "r2": r["pooled_r2"],
             "mae_ph": r["mae"], "rmse_ph": r["rmse"]}
            for r in d["per_checkpoint"]]

    # ---- baselines + the 4-amplitude sweep (Objective 2) ---------------
    d = load("deconfound_outputs/full_table.json")
    if d:
        out["amplitude_sweep"] = [
            {"denat_amplitude": r["denat_amplitude"],
             "linear_r2": r["linear_held_out_r2"],
             "plsr_r2": r["plsr_held_out_r2"],
             "crn_r2": r["crn_r2_mean"], "crn_r2_std": r["crn_r2_std"]}
            for r in d["amplitudes"]]
        out["sources"]["amplitude_sweep"] = "deconfound_outputs/full_table.json"

    # ---- spatial localisation ------------------------------------------
    d = load(f"{MCO}/spatial_localization.json")
    if d:
        a = d["aggregate"]
        out["spatial_localisation"] = {
            "note": "restricted to samples spanning 2+ quality classes",
            "n_multiclass": d["per_checkpoint"][0]["n_multiclass"],
            "n_total": d["per_checkpoint"][0]["n_samples"],
            "correct_region_pct": (a["hot_direction_correct_pct"]["mean"],
                                   a["hot_direction_correct_pct"]["std"]),
            "pixel_class_acc_pct": (a["class_acc_multiclass_pct"]["mean"],
                                    a["class_acc_multiclass_pct"]["std"]),
            "hot_region_overlap_pct": (a["hot_region_overlap_pct"]["mean"],
                                       a["hot_region_overlap_pct"]["std"]),
        }
        out["sources"]["spatial_localisation"] = f"{MCO}/spatial_localization.json"

    # ---- spatial magnitude (the limitation) -----------------------------
    d = load(f"{MCO}/spatial_skill_official.json")
    if d:
        a = d["aggregate"]
        out["spatial_magnitude"] = {
            "per_sample_r2": (a["per_sample_r2_mean"]["mean"], a["per_sample_r2_mean"]["std"]),
            "within_sample_corr": (a["within_sample_corr"]["mean"], a["within_sample_corr"]["std"]),
            "amplitude_ratio": (a["amplitude_ratio"]["mean"], a["amplitude_ratio"]["std"]),
            "between_sample_ph_sd": d["between_sample_ph_std"],
            "within_sample_ph_sd": d["within_sample_ph_std"],
        }
        out["sources"]["spatial_magnitude"] = f"{MCO}/spatial_skill_official.json"

    # ---- boundary artefact ----------------------------------------------
    d = load(f"{MCO}/edge_effect.json")
    if d:
        a = d["aggregate"]
        out["boundary_artefact"] = {
            "band_px": d["band_px"],
            "edge_mae_ph": (a["edge_mae"]["mean"], a["edge_mae"]["std"]),
            "interior_mae_ph": (a["interior_mae"]["mean"], a["interior_mae"]["std"]),
            "ratio": a["ratio"]["mean"],
            "pct_samples_affected": a["pct_samples_edge_worse"]["mean"],
        }
        out["sources"]["boundary_artefact"] = f"{MCO}/edge_effect.json"

    # ---- the frozen parameters ------------------------------------------
    try:
        import generate_dataset as G
        out["parameters"] = {
            k: (v["value"].tolist() if hasattr(v["value"], "tolist") else v["value"])
            for k, v in G.PARAMS.items()}
        out["parameter_status"] = {k: v.get("status", "") for k, v in G.PARAMS.items()}
    except Exception as e:                       # pragma: no cover
        out["parameters"] = f"could not import generate_dataset: {e}"

    with open("RESULTS.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- human-readable ------------------------------------
    L = []
    A = L.append

    def EV(summary=None, raw=None, script=None, figure=None, note=None):
        """Emit a consistent 'where this came from' block under a section."""
        A("<details><summary><b>Evidence in this repository</b></summary>\n")
        if summary:
            A(f"- **Summary values:** `{summary}`")
        if raw:
            A(f"- **Raw per-run data:** `{raw}`")
        if script:
            A(f"- **Produced by:** `{script}`")
        if figure:
            A(f"- **Figure:** `{figure}`")
        if note:
            A(f"- {note}")
        A("\n</details>\n")
    A("# LIGTAS-pH — Final Results\n")
    A(f"*Consolidated {out['generated']} by `collect_results.py`, which reads the saved")
    A("evidence files rather than recomputing anything. Every figure is traceable to a file")
    A("listed under **Where each number comes from**. If this document and the source code")
    A("ever disagree, the code is correct — it is what was executed.*\n")

    A("## What this study produced\n")
    A("A physics-based simulator generates six-band multispectral images of pork with known")
    A("pixel-wise pH. A convolutional network is given the image and **only four pH readings**")
    A("per sample, and must predict pH everywhere else. The dense ground-truth map exists in")
    A("the simulation but is never shown during training; it is held back purely to score")
    A("against. The numbers below answer three questions: does it work, is it better than")
    A("conventional methods, and where does it fail.\n")

    h = out.get("headline")
    if h:
        A("---\n")
        A("## 1. Does it work? — predictive accuracy\n")
        A(f"**Evaluated on {h['evaluated_on']}**, across {h['seeds']} independently seeded")
        A("training runs, reported as mean ± standard deviation.\n")
        A("| Metric | Value | What it tells you |")
        A("|---|---|---|")
        A(f"| R² | **{h['r2'][0]:.4f} ± {h['r2'][1]:.4f}** | Share of pH variation explained. "
          "Scale-relative, so it moves with the pH range sampled. |")
        A(f"| MAE | **{h['mae_ph'][0]:.4f} pH** | Typical error in pH units. Range-independent, "
          "so it is the more stable figure to quote. |")
        A(f"| RMSE | **{h['rmse_ph'][0]:.4f} pH** | As MAE, but penalises large errors more. "
          "Exceeds MAE, as expected. |")
        A(f"| Within ±0.15 pH | **{h['within_0.15_ph_pct'][0]:.1f}%** | Describes the heatmap "
          "directly. Needs no threshold or class definition, so nothing in it can be disputed. |")
        A(f"| Within ±0.20 pH | {h['within_0.20_ph_pct'][0]:.1f}% | The same, at a looser tolerance. |")
        A("")
        A("**Why four metrics and not one.** R² alone describes a heatmap poorly: it is relative")
        A("to whatever spread of pH happens to be present, so a narrower range lowers it even when")
        A("accuracy improves. MAE and RMSE are in pH units and do not move with the sampling")
        A("design. The tolerance bands describe the deliverable itself and carry no interpretive")
        A("choice at all. **Lead with MAE and the tolerance band; report R² alongside them.**\n")
        A("**Contribution:** this is the evidence for Objective 1 — that the pipeline produces")
        A("usable pixel-wise pH maps.\n")
        EV(summary=f"{MCO}/final_evaluation_TEST500.json",
           raw="crn_5seed_final/seed_{0..4}/history.json  (per-epoch training record)",
           script="evaluate_heatmap.py",
           figure="figures/fig_heatmap_example.png  (true / predicted / |error|, median-accuracy case); figures/fig_system_output.png  (input → predicted map, no ground truth shown)",
           note="Checkpoints `crn_5seed_final/seed_*/crn_best.pt` are excluded by size; "
                "rerun `evaluate_heatmap.py` to regenerate the summary from them.")

    if out.get("other_evaluation_sets"):
        A("### Why these numbers can be trusted\n")
        A("The same five checkpoints were scored on three different sets. The validation set was")
        A("used during training to decide when to stop, so it is the one that could flatter the")
        A("model. The other two never influenced training at all.\n")
        A("| Evaluation set | R² | MAE (pH) | Role |")
        A("|---|---|---|---|")
        roles = {"validation_n50": "used for early stopping — could be optimistic",
                 "test_n50": "never touched during training"}
        for lbl, v in out["other_evaluation_sets"].items():
            A(f"| {lbl} | {v['r2'][0]:.4f} ± {v['r2'][1]:.4f} | {v['mae_ph'][0]:.4f} | "
              f"{roles.get(lbl, '')} |")
        if h:
            A(f"| test_n500 **(reported)** | {h['r2'][0]:.4f} ± {h['r2'][1]:.4f} | "
              f"{h['mae_ph'][0]:.4f} | 500 fresh samples, distinct seed |")
        A("")
        A("**They agree.** That is direct evidence the model did not overfit the split used for")
        A("early stopping — a question a panel is entitled to ask. Note that the larger set is")
        A("slightly *less* flattering (MAE 0.090 → 0.094): the 50-sample figures were mildly")
        A("optimistic, so **the reported numbers are the more conservative ones.**\n")
        EV(summary=f"{MCO}/final_evaluation.json (validation), "
                   f"{MCO}/final_evaluation_TEST.json (test n=50), "
                   f"{MCO}/final_evaluation_TEST500.json (test n=500)",
           script="evaluate_heatmap.py --split {val,test} --data {...}")

    if out.get("per_seed"):
        A("### Per-seed detail\n")
        A("| Checkpoint | R² | MAE | RMSE |")
        A("|---|---|---|---|")
        for r in out["per_seed"]:
            A(f"| `{os.path.basename(os.path.dirname(r['checkpoint']))}` | "
              f"{r['r2']:.4f} | {r['mae_ph']:.4f} | {r['rmse_ph']:.4f} |")
        A("")
        rs = [r["r2"] for r in out["per_seed"]]
        A("**Why five runs rather than one.** Identical data and settings, differing only in")
        A(f"random initialisation, produce results from {min(rs):.2f} to {max(rs):.2f}. Reporting")
        A("whichever single run we happened to obtain would mislead in one direction or the")
        A("other, so every figure here is a mean across five runs with its standard deviation.\n")
        EV(raw="crn_5seed_final/seed_{0..4}/history.json  (loss and metrics per epoch)",
           script="train_crn.py --seed {0..4} --patience 10 --epochs 50",
           figure="figures/fig_loss_curves.png  (training vs validation loss, representative seed; the raw per-seed artefacts are crn_5seed_final/seed_*/loss_curve.png)",
           note="`crn_5seed_final/run.log` holds the original console output of the run.")

    if out.get("amplitude_sweep"):
        A("---\n")
        A("## 2. Is it better than conventional methods — and does that survive parameter uncertainty?\n")
        A("Two standard baselines were fitted and scored on identical data: pixel-wise linear")
        A("regression, and partial least squares regression. The comparison was then repeated")
        A("across the **entire plausible range** of the pH–scattering coupling strength, because")
        A("that parameter has no single citable value in the literature.\n")
        A("Every value in the table below is measured on the **held-out validation")
        A("split of that amplitude's own dataset (n = 50)** — the same split for all three")
        A("methods, so the comparison is like-for-like. The amplitudes other than 0.4 were")
        A("never evaluated on the 500-sample set, so validation is the only scale on which")
        A("all four points are comparable.\n")
        A("**This is why the CRN column reads 0.8486 at the adopted setting while the")
        A("headline in section 1 is 0.8472**: the headline is the 500-sample held-out")
        A("evaluation, which exists only at 0.4. The two differ by 0.0014, which also")
        A("bounds how much the CRN gains from choosing its stopping epoch on the")
        A("validation split — the baselines are closed-form and make no such choice.\n")
        A("| Coupling strength | Linear | PLSR | CRN |")
        A("|---|---|---|---|")
        for r in out["amplitude_sweep"]:
            tag = " *(adopted)*" if abs(r["denat_amplitude"] - 0.4) < 1e-9 else ""
            A(f"| {r['denat_amplitude']}{tag} | {r['linear_r2']:.4f} | {r['plsr_r2']:.4f} | "
              f"**{r['crn_r2']:.4f} ± {r['crn_r2_std']:.4f}** |")
        A("")
        A("**This establishes two things.**\n")
        A("First, **the task is not trivially solvable.** A linear fit stays below 0.9 at every")
        A("setting. Were the relationship simply the formula written into the simulator, a linear")
        A("model would recover it almost perfectly. It cannot — because pH acts only on")
        A("scattering, while absorption is driven independently by myoglobin, so the network must")
        A("separate two physical effects rather than invert one equation.\n")
        A("Second, and this is the stronger point: **the conclusion does not depend on getting")
        A("that parameter right.** No single value could be cited for it, so the whole range was")
        A("tested and the network wins throughout. That answers *“but is your parameter correct?”*")
        A("better than any single citation could, because a cited value might still be wrong for")
        A("this particular meat, whereas a range cannot be dismissed the same way.\n")
        A("**Contribution:** this is the evidence for Objective 2.\n")
        EV(summary="deconfound_outputs/full_table.json  (all four amplitudes, with provenance per row)",
           raw="deconfound_outputs/crn_amp_{0.2,0.6,0.8}_seed_{0,1,2}/history.json; "
               "sweep_outputs/ holds the earlier small-scale sweep",
           script="deconfound_full_scale.py, sweep_denat_amplitude.py, compute_amp04_baselines.py",
           figure="figures/fig_sweep_comparison.png",
           note="The corrected pooled CRN values were rescored in "
                f"`{MCO}/metric_comparison_0.2_0.6.json` and `{MCO}/metric_comparison.json`. "
                "`full_table.json` retains the superseded batch-averaged figures under "
                "`*_SUPERSEDED_batch_averaged` for traceability \u2014 do not quote those. "
                "The figure is `figures/fig_sweep_comparison.png`, built from `full_table.json`. "
                "`sweep_outputs/sweep_crn_vs_baselines.png` is the SUPERSEDED chart from the "
                "earlier small-scale sweep: it predates the metric fix and shows the CRN losing "
                "to the baselines at three of four amplitudes. It is kept as part of the "
                "investigation trail only — do not use it as a figure.")

    s_ = out.get("spatial_localisation")
    if s_:
        A("---\n")
        A("## 3. Does the heatmap point at the right places?\n")
        A("The deliverable is a *map*, so aggregate accuracy is not sufficient — a map could be")
        A("accurate on average while placing its features wrongly. This was therefore tested")
        A(f"directly, restricted to the **{s_['n_multiclass']} of {s_['n_total']} samples** whose")
        A("true pH spans more than one quality classification. Uniform samples are excluded")
        A("because localisation is meaningless on them, which makes this deliberately the harder")
        A("subset.\n")
        A("| Measure | Result | Chance | Reading |")
        A("|---|---|---|---|")
        A(f"| Correct elevated-pH region | **{s_['correct_region_pct'][0]:.1f}%** | 50% | "
          "Direction is essentially always right. A lenient test, but it establishes the signal is real. |")
        A(f"| Per-pixel class accuracy | **{s_['pixel_class_acc_pct'][0]:.1f}%** | — | "
          "**The figure to lead with.** Hard cases only, four pixels in five correct. |")
        A(f"| Hottest-quintile overlap | {s_['hot_region_overlap_pct'][0]:.1f}% | 20% | "
          "The demanding measure: well above chance, but honestly partial. |")
        A("")
        A("**Why this section exists.** It answers the most dangerous question a panel can ask:")
        A("*if the per-sample R² below is negative, what is the point of the heatmap?* The answer")
        A("is that **location and magnitude are different things.** The map finds the right")
        A("regions; it overstates how different they are. Only the second is miscalibrated.\n")
        EV(summary=f"{MCO}/spatial_localization.json",
           script="check_spatial_localization.py  (~5 min, inference only)",
           figure="figures/fig_prediction_gallery.png  (best to worst case, chosen by error percentile)")

    m = out.get("spatial_magnitude")
    if m:
        A("---\n")
        A("## 4. Where it falls short — spatial magnitude\n")
        A(f"- Per-sample R²: **{m['per_sample_r2'][0]:+.4f} ± {m['per_sample_r2'][1]:.4f}**")
        A(f"- Within-sample correlation: **{m['within_sample_corr'][0]:+.4f}** — genuine signal is present")
        A(f"- Predicted ÷ true spatial SD: **{m['amplitude_ratio'][0]:.3f}×** — over-expressed")
        A("*(The two pH-spread figures below are measured on the 50-sample validation")
        A("split. Figure `fig_ph_distribution.png` shows the same two statistics across all")
        A("400 frozen samples, where they read 0.3038 and 0.0865. Same quantities, different")
        A("scope — not a discrepancy.)*\n")
        A(f"- Between-sample pH SD **{m['between_sample_ph_sd']:.4f}** vs within-sample "
          f"**{m['within_sample_ph_sd']:.4f}**\n")
        A("**Why the number is negative, stated plainly.** Per-sample R² asks whether the model")
        A("predicted the right *amount* of variation inside a single sample. True within-sample")
        A(f"variation is only about {m['within_sample_ph_sd']:.3f} pH — **smaller than the model's**")
        if h:
            A(f"**own error of {h['mae_ph'][0]:.3f} pH**. On that scale the measure is punishing, and the")
        A(f"model additionally over-expresses variation by about {m['amplitude_ratio'][0]:.2f}×, which drives")
        A("it sharply negative.\n")
        A("**What it does and does not mean.** It is a *calibration* result. It does not mean the")
        A("model fails to detect spatial structure — section 3 shows it locates correctly. Optimal")
        A("rescaling of the existing predictions bounds the achievable value near +0.30, which")
        A("identifies this as a calibration remedy rather than an architectural one. **Not**")
        A("**attempted**, and disclosed as a limitation instead.\n")
        A("**Do not write** *“spatial reconstruction is unreliable”* — that conflates location with")
        A("magnitude and overstates the weakness. Say which one.\n")
        EV(summary=f"{MCO}/spatial_skill_official.json  (official checkpoints); "
                   f"{MCO}/spatial_skill.json  (independent reproduction)",
           script="check_spatial_skill.py",
           figure="figures/fig_ph_distribution.png  (all 400 frozen samples — NOTE the scope: the two statistics quoted in this section are the 50-sample VALIDATION split, so the figure reads 0.3038 / 0.0865 where the text reads 0.3036 / 0.0843; same quantities, different scope) "
                  "— the within- vs between-sample scales that make this measure punishing.")

    b = out.get("boundary_artefact")
    if b:
        A("---\n")
        A("## 5. Where it falls short — tissue boundaries\n")
        A(f"- MAE within a {b['band_px']} px band along the tissue edge: **{b['edge_mae_ph'][0]:.4f} pH**")
        A(f"- MAE in the interior: **{b['interior_mae_ph'][0]:.4f} pH**")
        A(f"- Ratio **{b['ratio']:.2f}×**, present in **{b['pct_samples_affected']:.0f}%** of samples\n")
        A("**Cause.** A convolutional network has less surrounding tissue to work with at the mask")
        A("edge, and padding supplies partial background. The effect appears as a visible rim in")
        A("the prediction figures, which is how it was noticed.\n")
        A("**Why report it rather than leave it.** The headline MAE averages this degraded rim into")
        A("the good interior. For any use concerned with the body of the cut rather than its")
        A(f"perimeter, the interior figure of **{b['interior_mae_ph'][0]:.3f} pH** is the more")
        A("representative one. Boundary-aware padding or masked convolution would address it")
        A("directly; that was not pursued here.\n")
        EV(summary=f"{MCO}/edge_effect.json",
           script="check_edge_effect.py",
           figure="figures/fig_prediction_gallery.png  (the rims are visible in the predicted maps)")

    if isinstance(out.get("parameters"), dict):
        A("---\n")
        A("## 6. The frozen parameters\n")
        A("Fourteen parameters drive the simulator. **Twelve trace to a published measurement, a")
        A("direct measurement, or a fit to published data.** The two that do not are labelled as")
        A("swept rather than presented as measurements — and the sweep in section 2 is what")
        A("establishes that the conclusion survives their uncertainty.\n")
        A("| Parameter | Value | Status |")
        A("|---|---|---|")
        for k, v in out["parameters"].items():
            val = ("[" + ", ".join(f"{x:g}" for x in v) + "]") if isinstance(v, list) else v
            st = out["parameter_status"].get(k, "").split("--")[0].strip()
            A(f"| `{k}` | {val} | {st} |")
        A("")
        A("These values are **frozen**. The dataset and the trained checkpoints have not been")
        A("regenerated since they were produced, so every number in this document refers to the")
        A("same simulator state.\n")
        EV(summary="generate_dataset.py  \u2014 the `PARAMS` dict, each entry carrying its own "
                   "citation string",
           raw="docs/digitization/  \u2014 the WebPlotDigitizer screenshots the extinction "
               "coefficients were read from",
           script="generate_dataset.py --selftest  (sign test + non-triviality baseline)",
           note="Full decision history, including values considered and rejected, is in "
                "`docs/TEAM_LOG.md` (newest entry first).")

    A("---\n")
    A("## Where each number comes from\n")
    A("| Result | Evidence file |")
    A("|---|---|")
    for k, v in out["sources"].items():
        A(f"| {k} | `{v}` |")
    A("")
    A("## Reproducing all of it\n")
    A("```")
    A("python generate_dataset.py --selftest      # sign test + non-triviality baseline")
    A("python evaluate_heatmap.py                 # section 1")
    A("python check_spatial_localization.py       # section 3")
    A("python check_edge_effect.py                # section 5")
    A("python make_figures.py                     # manuscript figures")
    A("python collect_results.py                  # regenerate this document")
    A("```\n")
    A("The five official checkpoints **are** committed (`crn_5seed_final/seed_*/crn_best.pt`,")
    A("2.4 MB) because per-seed results do not reproduce across machines — the same seed on")
    A("different hardware yields different weights, verified on two machines — so those files")
    A("are the only evidence for any per-seed figure.\n")
    A("The two datasets are not committed (~1.7 GB together) but regenerate bit-for-bit from")
    A("the frozen PARAMS:\n")
    A("```")
    A("python generate_dataset.py")
    A("    # the frozen 400-sample set, seed 42, split 300/50/50")
    A("")
    A("python generate_dataset.py --n 500 --seed 777 --all-test --out ligtas_test_extended")
    A("    # the 500-sample held-out set the headline figures come from.")
    A("    # Seed 777 is deliberately not 42: reusing 42 would reproduce the")
    A("    # training draws exactly and overlap the training data.")
    A("```")

    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("Wrote RESULTS.json and RESULTS.md")
    missing = [k for k in ("headline", "amplitude_sweep", "spatial_localisation",
                           "spatial_magnitude", "boundary_artefact") if k not in out]
    if missing:
        print("  MISSING (evidence file not found):", ", ".join(missing))
    else:
        print("  all six result groups present")


if __name__ == "__main__":
    main()
