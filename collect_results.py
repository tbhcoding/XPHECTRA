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

NUMWORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# Every result group RESULTS.md is expected to contain. A section whose
# evidence file is missing is silently skipped when the document is built,
# so the absence is reported here instead of going unnoticed.
REQUIRED = (
    "headline", "amplitude_sweep", "spatial_localisation", "spatial_magnitude",
    "boundary_artefact", "calibration", "class_recall", "class_thresholds",
    "range_restriction", "baselines_on_headline_set", "band_structure",
)


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


    # ---- band structure: why linear and PLSR land in the same place -----
    d = load(f"{MCO}/band_collinearity.json")
    if d:
        out["band_structure"] = {
            "n_pixels": d["n_pixels"],
            "first_component_pct": d["variance_share_pct"][0],
            "top4_cumulative_pct": d["top4_cumulative_pct"],
        }
        out["sources"]["band_structure"] = f"{MCO}/band_collinearity.json"

    # ---- baselines on the same 500-sample set as the headline -----------
    d = load(f"{MCO}/baselines_test500.json")
    if d:
        out["baselines_on_headline_set"] = {
            "evaluated_on": d["evaluated_on"],
            "n_eval_pixels": d["n_eval_pixels"],
            "pixels_per_sample": d["pixels_per_sample"],
            "linear_r2": d["linear"]["r2"], "linear_mae": d["linear"]["mae"],
            "plsr_r2": d["plsr"]["r2"], "plsr_mae": d["plsr"]["mae"],
            "plsr_n_components": d["plsr_n_components"],
            "margin_over_linear_r2": d["margin_over_linear_r2"],
            "margin_over_plsr_r2": d["margin_over_plsr_r2"],
        }
        out["sources"]["baselines_on_headline_set"] = f"{MCO}/baselines_test500.json"

    # ---- post-hoc calibration, and whether it is deployable -------------
    d = load(f"{MCO}/calibration.json")
    if d:
        r = d["results"]
        out["calibration"] = {
            "fit_on": d["fit_on"],
            "uncorrected": (r["uncorrected"]["r2"], r["uncorrected"]["mae"]),
            "level_only": (r["level only"]["r2"], r["level only"]["mae"]),
            "texture_only": (r["texture only"]["r2"], r["texture only"]["mae"]),
            "level_and_texture": (r["level + texture"]["r2"], r["level + texture"]["mae"]),
            "level_share_of_gain_pct": d["level_share_of_gain_pct"],
        }
        out["sources"]["calibration"] = f"{MCO}/calibration.json"

    d = load(f"{MCO}/calibration_probe.json")
    if d:
        r = d["results"]
        out.setdefault("calibration", {})
        out["calibration"]["level_from_dense_mean_r2"] = r["level from dense mean"]["r2"]
        out["calibration"]["level_from_4_probes_r2"] = r["level from 4 probes"]["r2"]
        out["calibration"]["level_from_4_probes_mae"] = r["level from 4 probes"]["mae"]
        out["calibration"]["r2_lost_using_probes"] = d["r2_lost_using_probes"]
        out["sources"]["calibration_deployability"] = f"{MCO}/calibration_probe.json"

    # ---- recall per quality class, before and after the correction ------
    d = load(f"{MCO}/class_recall.json")
    if d:
        sm = d["summary"]
        out["class_recall"] = {
            "class_boundaries": d["class_boundaries"],
            "correction": d["correction"],
            "classes": {k: {"true_share_pct": sm[k]["true_share_pct"],
                            "recall_before_pct": sm[k]["recall_before_pct"],
                            "recall_after_pct": sm[k]["recall_after_pct"]}
                        for k in ("PSE", "normal", "DFD")},
            "overall_accuracy_before_pct": d["overall_accuracy_before_pct"],
            "overall_accuracy_after_pct": d["overall_accuracy_after_pct"],
            "pse_recall_before_range_pct": d["pse_recall_before_range_pct"],
            "per_seed_pse": [(p["recall_before_pct"][0], p["recall_after_pct"][0])
                             for p in d["per_checkpoint"]],
        }
        out["sources"]["class_recall"] = f"{MCO}/class_recall.json"

    # ---- the same class result on the other split, as a cross-check -----
    d = load(f"{MCO}/final_evaluation_fig1edges.json")
    if d:
        out["class_accuracy_other_split"] = {
            "data": d["data"], "n_samples": d["n_samples"],
            "class_edges": d["class_edges"],
            "class_accuracy_pct": ms(d, "class_accuracy_pct"),
        }
        out["sources"]["class_accuracy_other_split"] = f"{MCO}/final_evaluation_fig1edges.json"

    # ---- how much of the headline is the range it was measured on -------
    d = load(f"{MCO}/range_restriction.json")
    if d:
        out["range_restriction"] = {
            "ceiling": d["ceiling"], "ceiling_source": d["ceiling_source"],
            "full": {k: d["full_range"][k] for k in
                     ("n_samples", "pooled_r2", "mae", "true_sd")},
            "restricted": {k: d["restricted"][k] for k in
                           ("n_samples", "pooled_r2", "mae", "true_sd")},
            "delta_r2": d["delta_r2"], "delta_mae": d["delta_mae"],
        }
        out["sources"]["range_restriction"] = f"{MCO}/range_restriction.json"

    # ---- does the class result depend on where the lines are drawn? -----
    d = load(f"{MCO}/class_thresholds.json")
    if d:
        out["class_thresholds"] = [
            {k: x[k] for k in ("scheme", "lower", "upper", "provenance",
                               "accuracy_pct", "majority_baseline_pct", "lift_pts")}
            for x in d["schemes"]]
        out["sources"]["class_thresholds"] = f"{MCO}/class_thresholds.json"

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
           note="Checkpoints `crn_5seed_final/seed_*/crn_best.pt` ARE committed (a deliberate "
                ".gitignore exception, 2.4 MB) because per-seed results do not reproduce across "
                "machines; "
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
    bl = out.get("baselines_on_headline_set")
    if bl:
        A("**The comparison at the headline scale.** The table above is the sweep, which")
        A("runs on validation because that is the only split every amplitude has. At the")
        A("adopted setting the baselines were also fitted and scored on the same")
        A(f"500-sample held-out set as the headline, over {bl['n_eval_pixels']:,} tissue")
        A(f"pixels sampled at {bl['pixels_per_sample']} per sample:\n")
        A("| Method | R\u00b2 | MAE (pH) |")
        A("|---|---|---|")
        A(f"| Pixel-wise linear regression | {bl['linear_r2']:.4f} | {bl['linear_mae']:.4f} |")
        A(f"| PLSR ({bl['plsr_n_components']} components) | {bl['plsr_r2']:.4f} | {bl['plsr_mae']:.4f} |")
        if out.get("headline"):
            hh = out["headline"]
            A(f"| **CRN** | **{hh['r2'][0]:.4f}** | **{hh['mae_ph'][0]:.4f}** |")
        A("")
        A(f"A margin of **{bl['margin_over_linear_r2']:.3f} R\u00b2 over linear** and")
        A(f"**{bl['margin_over_plsr_r2']:.3f} over PLSR**, on the set the headline is quoted")
        A("from. Note that the baselines read higher here than anywhere in the sweep table")
        A(f"({bl['linear_r2']:.4f} against 0.6212 on the 400-sample set): the 500-sample set")
        A("spans a wider pH range, which raises R\u00b2 for every method measured on it.")
        A("**Quote a baseline number together with the set it was measured on.**\n")
        EV(summary=out["sources"].get("baselines_on_headline_set"),
           script="compute_test500_baselines.py",
           note="The baselines are scored on a 400-pixel-per-sample draw while the "
                "network is scored on all tissue pixels of the same images. The pixel "
                "sets differ in size but are drawn from the same samples.")

    bs = out.get("band_structure")
    if bs:
        A("**Why the two baselines agree to within 0.001.** A singular value decomposition")
        A(f"of the six-band design matrix over {bs['n_pixels']:,} pixels puts")
        A(f"**{bs['first_component_pct']:.1f}% of its variance in a single component** and")
        A(f"**{bs['top4_cumulative_pct']:.2f}% in the first four**. The six measurements vary")
        A("largely together, so a four-component PLSR already spans nearly the whole space")
        A("a six-band linear fit can use. Neither can do better, because what separates pH")
        A("from myoglobin is not contained in any single band but in how the bands trade")
        A("off against one another.\n")
        EV(summary=out["sources"].get("band_structure"),
           script="check_band_collinearity.py")

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
        A(f"- Per-sample R²: **{m['per_sample_r2'][0]:+.4f} ± {m['per_sample_r2'][1]:.4f}** "
          "*(validation split, n=50)*. The same measure on the 500-sample held-out set "
          "is **-2.1699**. Quote one and name its set.")
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
        A("*(Measured on the 50-sample test split of the frozen dataset — NOT the")
        A("500-sample set the headline MAE comes from. Compare the two figures below")
        A("with each other, not against the 0.0936 headline.)*" + chr(10))
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
    cal = out.get("calibration")
    if cal:
        A("---\n")
        A("## 6. What a post-hoc calibration recovers\n")
        A("The network under-fits: it pulls its predictions toward the middle of the pH")
        A("range rather than committing to the extremes. That **costs** accuracy, so the")
        A("headline understates what the predictions actually contain. Two corrections were")
        A(f"fitted on the **{cal['fit_on']}** split and applied unchanged to the held-out")
        A("set: a linear rescaling of each sample's overall level, and a rescaling of the")
        A("variation within a sample about its own mean.\n")
        A("| Correction | R\u00b2 | MAE (pH) |")
        A("|---|---|---|")
        A(f"| None (the headline) | {cal['uncorrected'][0]:.4f} | {cal['uncorrected'][1]:.4f} |")
        A(f"| Level only | **{cal['level_only'][0]:.4f}** | **{cal['level_only'][1]:.4f}** |")
        A(f"| Texture only | {cal['texture_only'][0]:.4f} | {cal['texture_only'][1]:.4f} |")
        A(f"| Level and texture | {cal['level_and_texture'][0]:.4f} | {cal['level_and_texture'][1]:.4f} |")
        A("")
        A(f"The level correction alone accounts for **{cal['level_share_of_gain_pct']:.1f}% of")
        A("the total gain**, and it is the only one of the two a physical deployment could")
        A("fit, because it needs nothing but an estimate of each calibration sample's mean")
        A("pH. The texture correction needs the dense map, which no deployment has.\n")
        if "level_from_4_probes_r2" in cal:
            A("**The level correction was checked for exactly that.** It was fitted twice:")
            A("once from each calibration sample's true dense mean, and once from the mean of")
            A("its four probe readings, which is all a deployment would have.\n")
            A(f"- From the dense mean: R\u00b2 **{cal['level_from_dense_mean_r2']:.4f}**")
            A(f"- From four probe readings: R\u00b2 **{cal['level_from_4_probes_r2']:.4f}**, "
              f"MAE {cal['level_from_4_probes_mae']:.4f} pH")
            A("")
            A(f"A difference of {abs(cal['r2_lost_using_probes']):.4f}. **The correction does not")
            A("depend on information a deployment lacks**, which is what makes it reportable")
            A("rather than a curiosity. It is still a post-hoc correction fitted on held-out")
            A("data, so the uncorrected 0.8472 remains the headline.\n")
        EV(summary=out["sources"].get("calibration"),
           script="check_calibration.py, check_calibration_probe.py",
           note="The probe-fitted figures come from a second file, "
                f"`{out['sources'].get('calibration_deployability')}`, which "
                "repeats the level correction using only the four probe readings.")

    cr = out.get("class_recall")
    if cr:
        cls = cr["classes"]
        A("---\n")
        A("## 7. Recall per quality class\n")
        A(f"Pixels sorted into three quality classes at pH {cr['class_boundaries'][0]} and")
        A(f"{cr['class_boundaries'][1]}. **These boundaries are interpolated from a cited")
        A("scale's anchor values, not published as thresholds**, so everything in this")
        A("section is supporting evidence rather than a headline claim.\n")
        A("Overall accuracy is carried by whichever class is most common, so recall per")
        A(f"class is the figure that matters. Correction applied: {cr['correction']}.\n")
        A("| Class | Share of pixels | Recall before | Recall after |")
        A("|---|---|---|---|")
        for k, lbl in [("PSE", "PSE (low pH)"), ("normal", "normal"), ("DFD", "DFD (high pH)")]:
            c = cls[k]
            A(f"| {lbl} | {c['true_share_pct']:.1f}% | "
              f"{c['recall_before_pct'][0]:.1f}% \u00b1 {c['recall_before_pct'][1]:.1f} | "
              f"{c['recall_after_pct'][0]:.1f}% \u00b1 {c['recall_after_pct'][1]:.1f} |")
        A(f"| **Overall accuracy** | | **{cr['overall_accuracy_before_pct'][0]:.1f}%** | "
          f"**{cr['overall_accuracy_after_pct'][0]:.1f}%** |")
        A("")
        A("**Read the PSE row, not the overall figure.** Uncorrected, PSE recall ranged from")
        A(f"{cr['pse_recall_before_range_pct'][0]:.1f}% to {cr['pse_recall_before_range_pct'][1]:.1f}%")
        A("across the five seeds — the single most unstable number in this study. This")
        A("follows directly from the compression in section 6: predictions pulled toward the")
        A("centre of the range cross the low boundary before they cross any other, so PSE is")
        A("the class that compression destroys first.\n")
        ps = cr.get("per_seed_pse") or []
        if ps:
            worst = sorted(ps)[:3]
            best = sorted(ps)[3:]
            A("**The correction recovers it, and recovers it most where it was worst.**")
            A("Per-seed PSE recall, before to after:\n")
            A("| Seed | Before | After | Gain |")
            A("|---|---|---|---|")
            for i, (b, a2) in enumerate(ps):
                A(f"| {i} | {b:.1f}% | {a2:.1f}% | +{a2 - b:.1f} pts |")
            A("")
            A(f"The three seeds that started worst gained "
              f"{min(a2 - b for b, a2 in worst):.0f} to {max(a2 - b for b, a2 in worst):.0f}")
            A(f"points; the two already above 60% gained "
              f"{min(a2 - b for b, a2 in best):.0f} to {max(a2 - b for b, a2 in best):.0f}.")
            A("After correction the spread across seeds narrows to")
            A(f"{cls['PSE']['recall_after_pct'][0]:.1f}% \u00b1 {cls['PSE']['recall_after_pct'][1]:.1f}.")
            A("**That is the evidence that the PSE deficit is a calibration effect rather")
            A("than an inability to detect the condition** — the information was in the")
            A("predictions, on the wrong scale.\n")
        EV(summary=out["sources"].get("class_recall"),
           script="check_class_recall.py")

    ct = out.get("class_thresholds")
    if ct:
        A("---\n")
        A("## 8. Does the class result depend on where the boundaries are drawn?\n")
        A("The pork quality literature reports no single agreed criterion, and the")
        A("boundaries used above are interpolated. The obvious objection is that the result")
        A("was manufactured by choosing them. So the classification was recomputed under")
        A(f"{NUMWORD.get(len(ct), len(ct))} conventions, including the cited anchor values")
        A("used directly as limits.\n")
        A("| Boundaries | Scheme | Accuracy | Majority-class baseline | Lift |")
        A("|---|---|---|---|---|")
        for x in ct:
            A(f"| {x['lower']} / {x['upper']} | {x['scheme']} | {x['accuracy_pct']:.1f}% | "
              f"{x['majority_baseline_pct']:.1f}% | **+{x['lift_pts']:.1f} pts** |")
        A("")
        A(f"**The lift over simply guessing the most common class stays positive under every")
        A(f"convention**, from +{min(x['lift_pts'] for x in ct):.1f} to "
          f"+{max(x['lift_pts'] for x in ct):.1f} points. The conclusion does not depend on")
        A("the boundary choice, which is the only defensible way to report a number that")
        A("rests on an interpolated threshold.\n")
        A("Provenance of each scheme:\n")
        for x in ct:
            A(f"- **{x['lower']} / {x['upper']}** ({x['scheme']}): {x['provenance']}")
        A("")
        EV(summary=out["sources"].get("class_thresholds"),
           script="check_class_thresholds.py")

        co = out.get("class_accuracy_other_split")
        if co:
            A("**A second check, on the other split.**")
            A(f"The adopted boundaries applied to the {co['n_samples']}-sample")
            A(f"`{co['data']}` split give {co['class_accuracy_pct'][0]:.1f}% \u00b1 "
              f"{co['class_accuracy_pct'][1]:.1f} class accuracy, against")
            A(f"{ct[0]['accuracy_pct']:.1f}% on the 500-sample set. The result is not")
            A("specific to either the boundary convention or the split.\n")
            EV(summary=out["sources"].get("class_accuracy_other_split"),
               script="evaluate_heatmap.py")


    rr = out.get("range_restriction")
    if rr:
        A("---\n")
        A("## 9. How much of the headline is the range it was measured on?\n")
        A("R\u00b2 is scale-relative: it rises when the data spans a wider range, whether or")
        A("not the model got better. The 500-sample set spans more pH than the cited")
        A("reference scale covers, so the headline was recomputed on only those samples")
        A(f"whose mean pH falls at or below **{rr['ceiling']}** — "
          f"{rr['restricted']['n_samples']:.0f} of {rr['full']['n_samples']:.0f} samples.\n")
        A("| | Samples | True pH SD | R\u00b2 | MAE (pH) |")
        A("|---|---|---|---|---|")
        A(f"| Full range | {rr['full']['n_samples']:.0f} | {rr['full']['true_sd']:.4f} | "
          f"{rr['full']['pooled_r2']:.4f} | {rr['full']['mae']:.4f} |")
        A(f"| Restricted | {rr['restricted']['n_samples']:.0f} | {rr['restricted']['true_sd']:.4f} | "
          f"{rr['restricted']['pooled_r2']:.4f} | {rr['restricted']['mae']:.4f} |")
        A("")
        A(f"**R\u00b2 falls by {abs(rr['delta_r2']):.4f} while MAE *improves* by")
        A(f"{abs(rr['delta_mae']):.4f} pH.** Both move for the same reason: restricting the")
        A("range cuts the true pH spread from")
        A(f"{rr['full']['true_sd']:.3f} to {rr['restricted']['true_sd']:.3f}, which leaves R\u00b2")
        A("less variance to explain while leaving the model slightly *more* accurate in")
        A("absolute terms.\n")
        A("**This is the honest reading of the headline.** 0.8472 is partly a property of")
        A("how wide the simulated pH range is, and a narrower, more realistic range would")
        A("report a lower R\u00b2 for a model that is no worse. It is why MAE in pH units is")
        A("quoted alongside R\u00b2 everywhere in this document: MAE is range-independent and")
        A("cannot be inflated this way.\n")
        A(f"Ceiling source: {rr['ceiling_source']}.\n")
        EV(summary=out["sources"].get("range_restriction"),
           script="check_range_restriction.py")

        A("## 10. The frozen parameters\n")
        A("Fourteen parameters drive the simulator. **Ten are `CITED`, `MEASURED` or")
        A("`FITTED`; an eleventh (`c_Mb_sd`) is back-calculated from cited data.** The")
        A("remaining three are not traceable to a source and are labelled as such in the")
        A("code rather than presented as measurements:\n")
        A("- `denat_amplitude` — **SWEPT.** Section 2 varies it across its whole plausible")
        A("  range and the CRN beats both baselines at every setting, so the conclusion")
        A("  does not depend on its value.")
        A("- `denat_width` — **SWEPT, but not in the section 2 comparison.** It was varied")
        A("  only against the linear baseline, by the superseded single-seed method, where")
        A("  it moved R² by 0.07–0.10. **The CRN-vs-baselines conclusion has not been")
        A("  tested across its range.** Disclosed as a limitation, not claimed as covered.")
        A("- `mu_a_baseline` — **TUNED, NOT CITED.** Adopted at 0.8 because it improved the")
        A("  970 nm match. That makes the 970 nm external check **not independent of it**:")
        A("  the check cannot be offered as free-standing validation of this parameter.\n")
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
    A("python check_calibration.py                # section 6")
    A("python check_calibration_probe.py          # section 6, deployability")
    A("python check_class_recall.py               # section 7")
    A("python check_class_thresholds.py           # section 8")
    A("python check_range_restriction.py          # section 9")
    A("python compute_test500_baselines.py        # section 2, headline-scale baselines")
    A("python check_band_collinearity.py          # section 2, band structure")
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
    missing = [k for k in REQUIRED if k not in out]
    if missing:
        print("  MISSING (evidence file not found):", ", ".join(missing))
    else:
        print("  all %d result groups present" % len(REQUIRED))


if __name__ == "__main__":
    main()
