"""
tfr_stats.py  -  cluster-based permutation tests on single-trial TFR power, plus control analyses.
Works together with tfr_functions.py (imported as tfx).

Sections
  1. Trial groups and windows      build_groups, eligible_trials, window_report, auto_window_end
  2. Single-trial power            extract_single_trial_power, compute_pooled_baseline, apply_log_baseline
  3. The test                      permutation_test  (cluster mass = sum |Welch t|)
  4. Plan runner                   run_plan  -> <out_dir>/<reference>/<comparison>/ (figures, csv)
  5. Plots                         plot_plan_channels
  6. Reporting                     cluster_table, fdr_bh, overview, report_clusters
  7. Band-level controls           band_trial_curves, window_compare, timecourse_compare,
                                   lock_check, pre_tone_rt_corr, plot_curves
  8. Behaviour helpers             load_licks, lick_tables, reward_delay
  9. Sign-flip permutation test    sign_flip_test, run_sign_flip

The test in one paragraph
  Welch t per time-frequency pixel (A vs B, single trials). Pixels with |t| > t_crit
  (two-tailed p<0.05) only DEFINE clusters (8 neighbours, positive and negative separately).
  Cluster mass = sum of |t| inside the cluster. Null = biggest cluster mass of each of n_perm
  label shuffles (group sizes kept). Cluster p = (1 + #null >= observed) / (n_perm + 1).
  Per channel p_min = smallest cluster p; across the channels of one comparison p_min is
  FDR-corrected (Benjamini-Hochberg) -> p_fdr.
"""

import io
import os
import re
import json
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.ndimage import label as _cc_label, sum as _cc_sum
from scipy.stats import t as t_dist, spearmanr

import tfr_functions as tfx

STRUCT8 = np.ones((3, 3), dtype=int)          # 8-neighbour connectivity on the (freq x time) map
PLOT_WINDOWS = {"Press": (-0.6, 2.0), "Tone": (-1.2, 1.0), "Release": (-1.2, 1.0)}


# ======================================================================
# 1. TRIAL GROUPS AND WINDOWS
def build_groups(beh, early_long_min_hold=None):
    """
    All trial groups used in the analyses, keyed by (outcome, subtype):
      (correct|late|early, short|long|any)
    Touches (is_touch: paw on the lever, hold <= 0.1 s) are NOT early trials. trials.csv flags them
    early=1, so they are removed from every early group here and kept only as ("early", "touch")
    for a sanity check. Early sub-groups:
      (early, quick) hold < 0.5 s      (early, held) hold >= 0.5 s
      (early, real) = (early, any)     (early, short-real) / (early, long-real) = short / long
    ("early", "any"/"short"/"long") are therefore ALWAYS touch-free (60 / 14 / 46 trials in this session).

    early_long_min_hold : float or None
        If set, creates ("early", "long-filtered"): early long trials whose hold > this value (seconds).
        This keeps only early long trials that held long enough to be comparable to correct long.
    """
    touch_all = beh["is_touch"].astype(bool)
    g = {}
    for o in ("correct", "late", "early"):
        base = beh[(beh[o] == 1) & ~touch_all] if o == "early" else beh[beh[o] == 1]
        for t in ("short", "long"):
            g[(o, t)] = base[base[t] == 1]
        g[(o, "any")] = base
    e = g[("early", "any")]
    hold = e["release"] - e["press"]
    g[("early", "touch")] = beh[(beh["early"] == 1) & touch_all]
    g[("early", "real")] = e
    g[("early", "quick")] = e[hold < 0.5]
    g[("early", "held")] = e[hold >= 0.5]
    g[("early", "short-real")] = g[("early", "short")]
    g[("early", "long-real")] = g[("early", "long")]

    # filtered early long: only trials that held long enough
    if early_long_min_hold is not None:
        el = g[("early", "long")]
        el_hold = el["release"] - el["press"]
        g[("early", "long-filtered")] = el[el_hold > early_long_min_hold]
    return g


def group_label(key):
    return f"{key[0]} / {key[1]}"


def eligible_trials(trials, alignment, window, pre_tone=False, cross_release=False, tol=0.01,
                    allow_pre=False):
    """
    Keep only trials for which the whole `window` (s, relative to the alignment) is valid:
      - starts after the press;
      - ends before the release (unless cross_release=True, e.g. for outcome after the release);
      - pre_tone=True: ends before the tone (the 'waiting' period, Press alignment).
    The same rule is applied to both groups, so both are in the same situation.
    tol = 0.01 s (the tone delay is exactly 0.35 / 1.05 s).
    """
    hold = trials["release"] - trials["press"]
    tone_delay = (trials["tone"] - trials["press"]).where(trials["tone"] > 0, np.inf)
    off = {"Press": pd.Series(0.0, index=trials.index), "Tone": tone_delay, "Release": hold}[alignment]
    start, end = off + window[0], off + window[1]
    ok = np.isfinite(off)
    if not allow_pre:                                 # allow_pre=True: window may start before the press
        ok &= (start >= -tol)

    if not cross_release:
        ok &= (end <= hold + tol)
    if pre_tone:
        ok &= (end <= tone_delay + tol)
    return trials[ok]


def available_after(trials, alignment, pre_tone=False):
    """Seconds from the alignment until the trial ends (release), or until the tone if pre_tone."""
    hold = trials["release"] - trials["press"]
    tone_delay = (trials["tone"] - trials["press"]).where(trials["tone"] > 0, np.nan)
    if alignment == "Press":
        av = pd.concat([hold, tone_delay], axis=1).min(axis=1) if pre_tone else hold
    elif alignment == "Tone":
        av = hold - tone_delay                       # = RT for correct / late; NaN for early
    else:
        raise ValueError("Release alignment: post-release time is not limited by the trial")
    return av.dropna()


def window_report(groups, keys=None, quantiles=(0.05, 0.10, 0.25, 0.50)):
    """How much post-event time does each group have? q10 = 90 % of the trials reach this."""
    rows = []
    for key in (keys or list(groups)):
        for al, pt in (("Press", False), ("Press", True), ("Tone", False)):
            av = available_after(groups[key], al, pt)
            if len(av):
                rows.append({"group": group_label(key),
                             "alignment": al + (" (until tone)" if pt else ""), "n": len(av),
                             **{f"q{int(100 * q)}": round(float(av.quantile(q)), 3) for q in quantiles}})
    return pd.DataFrame(rows)


def auto_window_end(trials_a, trials_b, alignment, keep=0.9, pre_tone=False, cap=None):
    """Longest window end that >= `keep` of the trials of BOTH groups can still cover (floor to 50 ms)."""
    ends = [available_after(t, alignment, pre_tone).quantile(1 - keep) for t in (trials_a, trials_b)]
    end = np.floor(min(ends) * 20) / 20
    return float(min(end, cap) if cap else end)


# ======================================================================
# 2. SINGLE-TRIAL POWER (no averaging over trials, otherwise trials cannot be shuffled)
def extract_single_trial_power(ds_data, trials_df, channel_id, event_column, band_settings,
                               tmin, tmax, fs, keep_window=None, decim=10):
    """
    Morlet power of ONE channel for every trial (average=False).
    Returns power (trials, freqs, times) float32, times, freqs, kept (trial index labels).
    keep_window crops right away (memory); decim keeps every decim-th sample.
    """
    n_pre, n_post = round(fs * abs(tmin)), round(fs * tmax)
    n_total = n_pre + n_post
    arr, kept = [], []
    for tid, t_sec in trials_df[event_column].dropna().items():
        idx = int(t_sec * fs)
        trace = ds_data.get_traces(start_frame=idx - n_pre, end_frame=idx + n_post,
                                   channel_ids=[channel_id], return_in_uV=True)
        if trace.shape[0] == n_total:
            arr.append(trace[:, 0])
            kept.append(tid)
    if not arr:
        raise ValueError(f"No valid trials for event column '{event_column}'")
    info = mne.create_info(ch_names=[channel_id], sfreq=fs, ch_types="seeg")
    epochs = mne.EpochsArray(np.array(arr)[:, np.newaxis, :], info, tmin=tmin, verbose=False)
    per_band, times = [], None
    for band in band_settings:
        tfr = epochs.compute_tfr(method="morlet", freqs=band["freqs"], n_cycles=band["n_cycles"],
                                 use_fft=True, return_itc=False, average=False, decim=decim,
                                 n_jobs=1, verbose=False)
        data, times = tfr.data[:, 0, :, :], tfr.times
        if keep_window is not None:
            m = (times >= keep_window[0]) & (times <= keep_window[1])
            data, times = data[:, :, m], times[m]
        per_band.append(data.astype(np.float32))
    freqs = np.concatenate([b["freqs"] for b in band_settings])
    return np.concatenate(per_band, axis=1), times, freqs, kept


def compute_pooled_baseline(power, times, baseline_window):
    """Mean raw power per frequency inside the baseline window over ALL pooled trials -> (freqs, 1)."""
    m = (times >= baseline_window[0]) & (times <= baseline_window[1])
    return power[:, :, m].mean(axis=(0, 2), dtype=np.float64)[:, np.newaxis]


def apply_log_baseline(power, baseline_vals):
    """log10(power / baseline) for every single trial."""
    return np.log10(power / baseline_vals[np.newaxis].astype(np.float32)).astype(np.float32)


# ======================================================================
# 3. THE TEST: cluster-based permutation test on Welch t, cluster mass = sum |t|
def _welch_t(m_a, v_a, n_a, m_b, v_b, n_b):
    se = np.sqrt(np.maximum(v_a, 0) / n_a + np.maximum(v_b, 0) / n_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (m_a - m_b) / se
    return np.where(se > 0, t, np.nan)               # no variance -> NaN (never part of a cluster)


def _clusters(mask, t):
    """Label a boolean mask; returns (labels, one cluster mass = sum |t| per cluster)."""
    lab, n = _cc_label(mask, structure=STRUCT8)
    if n == 0:
        return lab, np.zeros(0)
    return lab, np.asarray(_cc_sum(np.nan_to_num(np.abs(t)), lab, np.arange(1, n + 1)), float)


def _max_cluster(t, t_crit):
    best = 0.0
    with np.errstate(invalid="ignore"):
        for mask in (t > t_crit, t < -t_crit):
            if mask.any():
                best = max(best, _clusters(mask, t)[1].max())
    return best


def permutation_test(data_a, data_b, n_perm=2000, pval=0.05, seed=0, batch=50):
    """
    data_a (nA, freqs, times), data_b (nB, freqs, times); nA and nB may differ.
    Returns a dict: t_map, real_diff (mean A - mean B, log10 ratio), clusters (list, each with
    sign / mass / size / p / mask), sig_corrected (union of clusters with p < pval),
    p_min, t_crit, null (max cluster mass of every shuffle).
    """
    n_a, n_b = data_a.shape[0], data_b.shape[0]
    n_tot, shape = n_a + n_b, data_a.shape[1:]
    if n_a < 2 or n_b < 2:
        raise ValueError("need at least 2 trials per group")
    X = np.concatenate([data_a, data_b]).reshape(n_tot, -1).astype(np.float64)
    X2 = X ** 2

    def t_for(WA, WB):                               # WA / WB: (k, n_tot) weights 1/nA, 1/nB
        m_a, m_b = WA @ X, WB @ X
        v_a = (WA @ X2 - m_a ** 2) * n_a / (n_a - 1)
        v_b = (WB @ X2 - m_b ** 2) * n_b / (n_b - 1)
        return _welch_t(m_a, v_a, n_a, m_b, v_b, n_b), m_a - m_b

    WA = np.zeros((1, n_tot)); WA[0, :n_a] = 1.0 / n_a
    WB = np.zeros((1, n_tot)); WB[0, n_a:] = 1.0 / n_b
    t_real, d_real = (x[0].reshape(shape) for x in t_for(WA, WB))
    t_crit = t_dist.ppf(1 - pval / 2, n_tot - 2)

    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for s in range(0, n_perm, batch):
        k = min(batch, n_perm - s)
        order = np.argsort(rng.random((k, n_tot)), axis=1)
        rows = np.arange(k)[:, None]
        WA = np.zeros((k, n_tot)); WB = np.zeros((k, n_tot))
        WA[rows, order[:, :n_a]] = 1.0 / n_a
        WB[rows, order[:, n_a:]] = 1.0 / n_b
        T, _ = t_for(WA, WB)
        for i in range(k):
            null[s + i] = _max_cluster(T[i].reshape(shape), t_crit)

    clusters, sig = [], np.zeros(shape, bool)
    with np.errstate(invalid="ignore"):
        for sign, mask in ((1, t_real > t_crit), (-1, t_real < -t_crit)):
            if mask.any():
                lab, mass = _clusters(mask, t_real)
                for k, v in enumerate(mass, 1):
                    p = (1 + np.sum(null >= v)) / (n_perm + 1)
                    blob = lab == k
                    clusters.append({"sign": sign, "mass": float(v), "size": int(blob.sum()),
                                     "p": float(p), "mask": blob})
                    if p < pval:
                        sig |= blob
    return {"t_map": t_real, "real_diff": d_real, "clusters": clusters, "sig_corrected": sig,
            "p_min": min([c["p"] for c in clusters], default=1.0), "t_crit": t_crit,
            "null": null, "n_a": n_a, "n_b": n_b, "n_perm": n_perm, "pval": pval}


# ======================================================================
# 4. PLAN RUNNER
def slugify(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def fdr_bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def _event_mark(trials, alignment):
    """Median time (s) of the other task event relative to the alignment -> red dashed line."""
    has_tone = trials["tone"] > 0
    if has_tone.mean() <= 0.5:
        return None
    d = (trials["tone"] - trials["press"])[has_tone].median()
    if alignment == "Press":
        return float(d)
    if alignment == "Tone":
        return -float(d)
    return float((trials["tone"] - trials["release"])[has_tone].median())


def resolve_plan(plan, groups):
    """Fill in automatic window ends (window = (t0, None)) and add a slug to every item."""
    out = []
    for it in plan:
        it = dict(it)
        if it["window"][1] is None:
            end = auto_window_end(groups[it["A"]], groups[it["B"]], it["alignment"],
                                  keep=it.get("keep", 0.9), pre_tone=it.get("pre_tone", False),
                                  cap=it.get("cap"))
            it["window"] = (it["window"][0], end)
        it["slug"] = slugify(it["name"])
        out.append(it)
    return out


MAX_PATH = 250      # Windows stops at ~260 characters


def check_paths(items, out_dir, refs):
    """Fail BEFORE the long run if a result file path would be too long for Windows."""
    bad = []
    for it in items:
        for ref in refs:
            p = os.path.join(out_dir, ref, it["slug"], f"tfr_{ref}_part10.png")   # the longest file name
            if len(p) > MAX_PATH:
                bad.append((len(p), it["name"]))
    if bad:
        raise ValueError("path too long (Windows limit ~260): "
                         + "; ".join(f"'{n}' = {l} chars" for l, n in sorted(set(bad), reverse=True))
                         + ". Give these items a shorter plan name (item['name'] = ...).")


def run_plan(recordings, CONFIG, fs, plan, groups, all_trials, selected_df, probe_df, out_dir,
             contour=None, n_perm=2000, pval=0.05, seed=0, decim=10, min_n=10, chunk=6,
             freq_range=None, show=False, skip_done=True):
    """
    Cluster test for every plan item, every reference and every selected channel.

    plan       list of dicts: name, A, B (group keys), alignment, window (t0, t1 | None = automatic),
               pre_tone, cross_release, keep, cap, purpose
    recordings {"original": ds, "CAR": ds_car}   (one test per reference)

    Output:  out_dir/<reference>/<comparison>/
               tfr_<reference>_part<k>.png      A | B | A-B per channel, clusters outlined
               summary_<reference>.csv          one row per channel (p_min, p_fdr, ...)
               clusters_<reference>.csv         clusters with p <= 0.10
               item_info.json                              window actually used, n_perm, seed, ...
             out_dir/summary_all.csv, clusters_all.csv     everything of this call
    The single-trial power of each channel is extracted ONCE per alignment and shared by all items.
    Items whose summary csv already exists are skipped (skip_done=True) -> a crashed run can resume.
    """
    os.makedirs(out_dir, exist_ok=True)
    items = resolve_plan(plan, groups)
    check_paths(items, out_dir, list(recordings))
    freq_range = freq_range or CONFIG["zoom_freq_range"]
    base_ev = CONFIG["baseline_event"]
    aligns = sorted({it["alignment"] for it in items} | {base_ev})
    cols = CONFIG["events"]
    ev_trials = {al: all_trials.dropna(subset=[cols[al]]) for al in aligns}
    pwin = {al: PLOT_WINDOWS[al] for al in aligns}
    pwin[base_ev] = (min(pwin[base_ev][0], CONFIG["baseline_window"][0]),
                     max(pwin[base_ev][1], CONFIG["baseline_window"][1]))
    all_sum, all_clu = [], []

    for ref, ds in recordings.items():
        def folder(it):
            return os.path.join(out_dir, ref, it["slug"])

        def f_sum(it):
            return os.path.join(folder(it), f"summary_{ref}.csv")

        todo = [it for it in items if not (skip_done and os.path.exists(f_sum(it)))]
        for it in items:
            if it not in todo:
                all_sum.append(pd.read_csv(f_sum(it)))
                c = os.path.join(folder(it), f"clusters_{ref}.csv")
                if os.path.exists(c):
                    all_clu.append(pd.read_csv(c))
                print(f"[{ref}] {it['name']}: already done, loaded")
        if not todo:
            continue

        res = {it["slug"]: {"ch": {}, "rows": [], "clu": [], "times": None, "freqs": None,
                            "labels": None} for it in todo}
        for ch in selected_df["channel_id"]:
            print(f"[{ref}] {ch}: extracting power", flush=True)
            pw = {}
            with contextlib.redirect_stdout(io.StringIO()):
                for al in aligns:
                    pw[al] = extract_single_trial_power(
                        ds, ev_trials[al], ch, cols[al], CONFIG["band_settings"], CONFIG["tmin"],
                        CONFIG["tmax"], fs, keep_window=pwin[al], decim=decim)
            p_pow, p_times, freqs, p_kept = pw[base_ev]
            base = compute_pooled_baseline(p_pow, p_times, CONFIG["baseline_window"])
            for it in todo:
                ev, win, R = it["alignment"], it["window"], res[it["slug"]]
                power, times_full, freqs, kept = pw[ev]
                pos = {t: i for i, t in enumerate(kept)}
                d, maps, used = [], [], []
                try:
                    for key in (it["A"], it["B"]):
                        tr = eligible_trials(groups[key], ev, win, it.get("pre_tone", False),
                                             it.get("cross_release", False))
                        rows = [pos[t] for t in tr.index if t in pos]
                        if len(rows) < min_n:
                            raise ValueError(f"{group_label(key)}: {len(rows)} eligible trials (min {min_n})")
                        pr = power[rows]
                        m = (times_full >= win[0] - 1e-9) & (times_full <= win[1] + 1e-9)
                        d.append(apply_log_baseline(pr, base)[:, :, m])
                        maps.append(np.log10(pr.mean(axis=0) / base))
                        used.append(tr.loc[[t for t in tr.index if t in pos]])
                except ValueError as e:
                    print(f"   {it['name']}: skipped ({e})")
                    continue
                ft = it.get("freq_test")
                if ft:
                    fm = (freqs >= ft[0] - 1e-9) & (freqs <= ft[1] + 1e-9)
                    d = [x[:, fm, :] for x in d]
                    f_test = freqs[fm]
                else:
                    fm, f_test = None, freqs
                r = permutation_test(d[0], d[1], n_perm, pval, seed)
                t_test = times_full[m]
                sig_full = r["sig_corrected"]
                if ft:                                   # put the band mask back on the full frequency axis
                    sig_full = np.zeros((len(freqs), len(t_test)), bool)
                    sig_full[fm] = r["sig_corrected"]

                R["times"], R["freqs"] = times_full, freqs
                R["labels"] = (group_label(it["A"]), group_label(it["B"]))
                R["ch"][ch] = {"A": maps[0], "B": maps[1], "sig": sig_full, "sig_t": t_test,
                               "marks": (_event_mark(used[0], ev), _event_mark(used[1], ev)),
                               "nA": r["n_a"], "nB": r["n_b"]}
                R["rows"].append({
                    "comparison": it["name"], "reference": ref, "alignment": ev,
                    "window": f"{win[0]} to {win[1]}", "channel": ch, "nA": r["n_a"], "nB": r["n_b"],
                    "keepA %": round(100 * r["n_a"] / len(groups[it["A"]])),
                    "keepB %": round(100 * r["n_b"] / len(groups[it["B"]])),
                    "p_min": r["p_min"],
                    "sig clusters": int(sum(c["p"] < pval for c in r["clusters"])),
                    "biggest t-mass": max([c["mass"] for c in r["clusters"]], default=0.0)})
                if ft:                                   # effect size of the tested band x window (A vs B, %)
                    R["rows"][-1]["band diff % (A vs B)"] = round(
                        100 * (10 ** float(np.nanmean(r["real_diff"])) - 1), 1)
                    R["rows"][-1]["band mean t"] = round(float(np.nanmean(r["t_map"])), 2)
                R["clu"].append(cluster_table(r, t_test, f_test, R["labels"], it, ch, ref,
                                              CONFIG["band_ranges"]))
                print(f"   {it['name']}: nA={r['n_a']} nB={r['n_b']} p_min={r['p_min']:.4f}", flush=True)

        # ---- write figures + tables of every item of this reference ----
        for it in todo:
            R = res[it["slug"]]
            if not R["rows"]:
                continue
            os.makedirs(folder(it), exist_ok=True)
            sm = pd.DataFrame(R["rows"])
            sm["p_fdr"] = fdr_bh(sm["p_min"].values)          # across the channels of this comparison
            cl = pd.concat(R["clu"], ignore_index=True)
            sm.to_csv(f_sum(it), index=False)
            cl.to_csv(os.path.join(folder(it), f"clusters_{ref}.csv"), index=False)
            with open(os.path.join(folder(it), "item_info.json"), "w") as f:
                json.dump({"name": it["name"], "A": group_label(it["A"]), "B": group_label(it["B"]),
                           "alignment": it["alignment"], "window": it["window"],
                           "pre_tone": it.get("pre_tone", False),
                           "cross_release": it.get("cross_release", False),
                           "purpose": it.get("purpose", ""), "reference": ref, "n_perm": n_perm,
                           "seed": seed, "pval": pval, "decim": decim}, f, indent=2)
            sel = selected_df[selected_df["channel_id"].isin(R["ch"])].reset_index(drop=True)
            dvmax = float(np.nanpercentile(np.concatenate(
                [np.abs(v["A"] - v["B"]).ravel() for v in R["ch"].values()]), 99))
            for i in range(0, len(sel), chunk):
                k = i // chunk + 1
                plot_plan_channels(
                    R["ch"], R["times"], R["freqs"], R["labels"], it["alignment"], sel.iloc[i:i + chunk],
                    probe_df, title=f"[{ref}] {it['name']} | {it['alignment']} {it['window']} | "
                                    f"cluster test on t (mass), part {k}",
                    freq_range=freq_range, yticks=CONFIG["zoom_yticks"],
                    band_boundaries=CONFIG["oscillation_boundaries"], vmax=CONFIG["zoom_vmax"],
                    diff_vmax=dvmax, contour=contour,
                    save_path=os.path.join(folder(it), f"tfr_{ref}_part{k}.png"), show=show)
            all_sum.append(sm)
            all_clu.append(cl)

    summary = pd.concat(all_sum, ignore_index=True) if all_sum else pd.DataFrame()
    clusters = pd.concat(all_clu, ignore_index=True) if all_clu else pd.DataFrame()
    summary.to_csv(os.path.join(out_dir, "summary_all.csv"), index=False)
    clusters.to_csv(os.path.join(out_dir, "clusters_all.csv"), index=False)
    return summary, clusters


# ======================================================================
# 5. PLOT: per channel [A] [B] [A-B] [probe], clusters outlined
def _outline(ax, times, freqs, mask, color="white", lw=2.0):
    """Closed outline of a boolean mask (padded so clusters at the window edge are closed)."""
    if mask is None or not mask.any() or len(times) < 2:
        return
    f = np.asarray(freqs, float)
    dt = float(np.median(np.diff(times)))
    tp = np.r_[times[0] - dt, times, times[-1] + dt]
    fp = np.r_[f[0] - (f[1] - f[0]), f, f[-1] + (f[-1] - f[-2])]
    ax.contour(tp, fp, np.pad(mask.astype(float), 1), levels=[0.5], colors=color, linewidths=lw)


def plot_plan_channels(ch_res, times, freqs, labels, event, selected_df, probe_df, title,
                       freq_range=(1, 15), yticks=None, band_boundaries=None, vmax=0.5,
                       diff_vmax=None, contour=None, save_path=None, show=False, dpi=150):
    """
    Rows = channels. Columns: A (full TFR) | B (full TFR) | A-B | probe. White outline = cluster
    significant after the permutation test (only inside the tested window). Red dashed = other task
    event. A-B is the plain average difference (the test itself runs on single-trial t).
    ch_res[ch] = {"A","B","sig","sig_t","marks","nA","nB"}
    """
    yticks = yticks or tfx.DEFAULT_YTICKS
    chans = list(selected_df["channel_id"])
    n = len(chans)
    if diff_vmax is None:
        diff_vmax = float(np.nanpercentile(np.concatenate(
            [np.abs(ch_res[c]["A"] - ch_res[c]["B"]).ravel() for c in chans]), 99))
    cols_panel = [0, 2, 6]
    fig_h = 3.2 * n + 1.8
    fig = plt.figure(figsize=(19.5, fig_h))
    gs = fig.add_gridspec(n, 11, wspace=0.0, hspace=0.45,
                          width_ratios=[1, 0.10, 1, 0.06, 0.035, 0.34, 1, 0.06, 0.035, 0.42, 0.7],
                          left=0.06, right=0.97, top=1 - 1.4 / fig_h, bottom=0.6 / fig_h)
    im_ab = im_d = None
    for r, (_, row) in enumerate(selected_df.iterrows()):
        ch, d = row["channel_id"], ch_res[row["channel_id"]]
        panels = [(d["A"], vmax, d["marks"][0], "A"), (d["B"], vmax, d["marks"][1], "B"),
                  (d["A"] - d["B"], diff_vmax, None, "A - B")]
        for k, (arr, lim, mark, name) in enumerate(panels):
            ax = fig.add_subplot(gs[r, cols_panel[k]])
            im = ax.pcolormesh(times, freqs, arr, cmap="viridis", vmin=-lim, vmax=lim, shading="gouraud")
            _outline(ax, d["sig_t"], freqs, d["sig"])
            ax.axvline(0, color="white", linestyle="--", linewidth=1.2)
            if mark is not None:
                ax.axvline(mark, color="red", linestyle="--", linewidth=1.5)
            for f in (band_boundaries or []):
                ax.axhline(f, color="white", linestyle=":", linewidth=0.5)
            ax.set_ylim(freq_range); ax.set_yticks(yticks); ax.set_xlim(times[0], times[-1])
            ax.grid(False); ax.tick_params(labelsize=7); ax.set_xlabel("Time (s)", fontsize=8)
            if k == 0:
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.text(-0.22, 0.5, ch, transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=13, fontweight="bold")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                ax.set_title({"A": f"A: {labels[0]} (n={d['nA']})", "B": f"B: {labels[1]} (n={d['nB']})",
                              "A - B": "A - B"}[name], fontsize=12, fontweight="bold", pad=8)
            if k < 2:
                im_ab = im
            else:
                im_d = im
        axp = fig.add_subplot(gs[r, 10])
        tfx.draw_probe(axp, probe_df, row["x"], row["y"], contour)
        if r == 0:
            axp.set_title("Probe location", fontsize=12, fontweight="bold", pad=8)
    cb1 = fig.colorbar(im_ab, cax=fig.add_subplot(gs[:, 4]), extend="both")
    tfx.percent_ticks(cb1, vmax)
    cb1.set_label("Power change vs. baseline (%)", fontsize=9, labelpad=8)
    cb1.ax.tick_params(labelsize=7)
    cb2 = fig.colorbar(im_d, cax=fig.add_subplot(gs[:, 8]), extend="both")
    tfx.percent_ticks(cb2, diff_vmax, pct=(-60, -40, -20, 0, 25, 50, 100, 200, 400))
    cb2.set_label("A vs. B (%): yellow = A higher, purple = B higher", fontsize=9, labelpad=8)
    cb2.ax.tick_params(labelsize=7)
    fig.suptitle(title + "\nwhite outline = cluster-corrected significant (p < 0.05) | "
                 "red dashed = other task event", fontsize=15, fontweight="bold",
                 y=1 - 0.2 / fig_h, va="top")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ======================================================================
# 6. REPORTING
def cluster_table(r, times, freqs, labels, item, channel, ref, band_ranges, p_max=0.10):
    """One row per cluster of one test with p <= p_max (significant = p < pval)."""
    base = {"comparison": item["name"], "reference": ref, "channel": channel,
            "alignment": item["alignment"], "window": f"{item['window'][0]} to {item['window'][1]}",
            "nA": r["n_a"], "nB": r["n_b"]}
    t, diff, rows = np.nan_to_num(r["t_map"]), r["real_diff"], []
    for c in r["clusters"]:
        if c["p"] > p_max:
            continue
        ff, tt = np.where(c["mask"])
        hit = [b for b, (lo, hi) in band_ranges.items() if ((freqs[ff] >= lo) & (freqs[ff] <= hi)).any()]
        rows.append({**base, "cluster": len(rows) + 1,
                     "direction": f"{labels[0]} higher" if c["sign"] > 0 else f"{labels[1]} higher",
                     "t_start (s)": round(float(times[tt.min()]), 3), "t_end (s)": round(float(times[tt.max()]), 3),
                     "f_min (Hz)": float(freqs[ff.min()]), "f_max (Hz)": float(freqs[ff.max()]),
                     "bands": "+".join(hit), "size (pixels)": c["size"], "t-mass": round(c["mass"], 1),
                     "peak t": round(float(t[c["mask"]][np.argmax(np.abs(t[c["mask"]]))]), 2),
                     "mean change A vs B (%)": round(100 * (10 ** float(diff[c["mask"]].mean()) - 1), 1),
                     "cluster p": round(c["p"], 4), "significant": c["p"] < r["pval"]})
    return pd.DataFrame(rows) if rows else pd.DataFrame([{**base, "cluster": 0, "direction": "none"}])


def overview(summary, alpha=0.05):
    """One row per comparison x reference: how many channels significant (p_min<.05 / p_fdr<.05)."""
    g = summary.groupby(["comparison", "reference", "alignment", "window"], sort=False)
    return g.agg(n_channels=("channel", "nunique"), nA=("nA", "median"), nB=("nB", "median"),
                 n_p_min=("p_min", lambda s: int((s < alpha).sum())),
                 n_p_fdr=("p_fdr", lambda s: int((s < alpha).sum()))).reset_index()


def report_clusters(clusters, comparison=None, reference=None, only_sig=True):
    """Compact text list of clusters (for the notes)."""
    c = clusters
    if comparison:
        c = c[c["comparison"] == comparison]
    if reference:
        c = c[c["reference"] == reference]
    c = c[c["cluster"] > 0]
    if only_sig:
        c = c[c["significant"].astype(bool)]
    cols = ["comparison", "reference", "channel", "direction", "t_start (s)", "t_end (s)", "f_min (Hz)",
            "f_max (Hz)", "bands", "mean change A vs B (%)", "peak t", "cluster p"]
    return c[cols].sort_values(["comparison", "reference", "channel"]).reset_index(drop=True)


# ======================================================================
# 7. BAND-LEVEL CONTROLS (one number per trial = mean over the selected channels)
class BandCurves:
    """Single-trial band power curves (log10 ratio vs pooled pre-Press baseline), channel mean."""
    def __init__(self, times, data, index, label=""):
        self.times, self.data, self.index, self.label = times, data, list(index), label
        self._pos = {t: i for i, t in enumerate(self.index)}

    def get(self, trials):
        """Rows of the trials (a DataFrame subset) that exist in this object -> (n, T)."""
        return self.data[[self._pos[t] for t in trials.index if t in self._pos]]


def band_trial_curves(ds_data, trials_all, channel_ids, event_col, spec, fs, baseline_window,
                      keep_window, tmin=-1.5, tmax=2.5, decim=10, press_col="press_np_s", label=""):
    """
    Band power per trial around `event_col`, averaged over frequencies of `spec` and over channels.
    spec = {"freqs": np.arange(5, 9), "n_cycles": 2}  (short wavelets -> little smearing in time)
    The baseline is the pooled pre-Press window of ALL trials, per channel.
    """
    ev_trials = trials_all.dropna(subset=[event_col])
    per_ch, kept0, times = [], None, None
    for ch in channel_ids:
        bp, bt, _, _ = extract_single_trial_power(ds_data, trials_all.dropna(subset=[press_col]), ch,
                                                  press_col, [spec], tmin, tmax, fs,
                                                  keep_window=baseline_window, decim=decim)
        base = compute_pooled_baseline(bp, bt, baseline_window)
        pw, times, _, kept = extract_single_trial_power(ds_data, ev_trials, ch, event_col, [spec],
                                                        tmin, tmax, fs, keep_window=keep_window, decim=decim)
        if kept0 is None:
            kept0 = kept
        assert kept == kept0, "different trials kept in different channels"
        per_ch.append(apply_log_baseline(pw, base).mean(axis=1))      # mean over frequencies
    return BandCurves(times, np.mean(per_ch, axis=0), kept0, label)


def _pct(x):
    return 100 * (10 ** x - 1)


def _t(a, b):
    """Welch t per column of a (nA, k), b (nB, k)."""
    return _welch_t(a.mean(0), a.var(0, ddof=1), len(a), b.mean(0), b.var(0, ddof=1), len(b))


def _perm_t(a, b, n_perm, rng):
    X, n_a = np.vstack([a, b]), len(a)
    out = np.empty((n_perm, a.shape[1]))
    for k in range(n_perm):
        idx = rng.permutation(len(X))
        out[k] = _t(X[idx[:n_a]], X[idx[n_a:]])
    return out


def window_compare(bc, A, B, window, n_boot=5000, n_perm=5000, seed=0):
    """
    One number per trial = mean of the curve inside `window`. A, B = DataFrames of eligible trials
    (use eligible_trials). Returns a dict: n, means (%), difference A-B (%), bootstrap 95 % CI, perm p.
    """
    rng = np.random.default_rng(seed)
    m = (bc.times >= window[0] - 1e-9) & (bc.times <= window[1] + 1e-9)
    a, b = bc.get(A)[:, m].mean(1)[:, None], bc.get(B)[:, m].mean(1)[:, None]
    diff = float(a.mean() - b.mean())
    boot = [rng.choice(a[:, 0], len(a)).mean() - rng.choice(b[:, 0], len(b)).mean() for _ in range(n_boot)]
    t_obs = _t(a, b)
    t_null = _perm_t(a, b, n_perm, rng)
    p = (1 + np.sum(np.abs(t_null) >= np.abs(t_obs))) / (n_perm + 1)
    return {"nA": len(a), "nB": len(b), "A (%)": round(float(_pct(a.mean())), 1), "B (%)": round(float(_pct(b.mean())), 1),
            "A-B (log10)": round(diff, 4), "A vs B (%)": round(_pct(diff), 1),
            "CI low (%)": round(float(_pct(np.percentile(boot, 2.5))), 1),
            "CI high (%)": round(float(_pct(np.percentile(boot, 97.5))), 1), "t": round(float(t_obs[0]), 2),
            "p": round(float(p), 4)}


def timecourse_compare(bc, A, B, window, bin_s=0.05, n_perm=5000, n_boot=2000, seed=0):
    """
    Same comparison in consecutive bins of `bin_s` seconds. p (raw) per bin; p (max) corrects across
    the bins with the maximum |t| over bins of every label shuffle.
    """
    rng = np.random.default_rng(seed)
    edges = np.arange(window[0], window[1] + 1e-9, bin_s)
    ba, bb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (bc.times >= lo - 1e-9) & (bc.times < hi - 1e-9)
        ba.append(bc.get(A)[:, m].mean(1))
        bb.append(bc.get(B)[:, m].mean(1))
    a, b = np.array(ba).T, np.array(bb).T                     # (trials, bins)
    t_obs = _t(a, b)
    t_null = np.abs(_perm_t(a, b, n_perm, rng))
    p_raw = (1 + (t_null >= np.abs(t_obs)).sum(0)) / (n_perm + 1)
    p_max = (1 + (t_null.max(1)[:, None] >= np.abs(t_obs)).sum(0)) / (n_perm + 1)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    boot = a[ia].mean(1) - b[ib].mean(1)                      # (n_boot, bins)
    diff = a.mean(0) - b.mean(0)
    return pd.DataFrame({
        "bin start (s)": edges[:-1].round(3), "bin end (s)": edges[1:].round(3),
        "A (%)": _pct(a.mean(0)).round(1), "B (%)": _pct(b.mean(0)).round(1),
        "A vs B (%)": _pct(diff).round(1),
        "CI low (%)": _pct(np.percentile(boot, 2.5, axis=0)).round(1),
        "CI high (%)": _pct(np.percentile(boot, 97.5, axis=0)).round(1),
        "t": t_obs.round(2), "p (raw)": p_raw.round(4), "p (max over bins)": p_max.round(4),
        "sig": np.where(p_max < 0.05, "*", ""), "nA": len(a), "nB": len(b)})


def lock_check(bc_tone, bc_release, trials, search_tone=(0.0, 0.9), search_release=(-0.2, 0.6),
               n_groups=3, smooth=5):
    """
    Is a dip locked to the TONE or to the RELEASE? Split correct/late trials into RT groups
    (fast ... slow), find the time of the lowest value of the mean curve in every group, once
    relative to the tone and once relative to the release. The alignment in which the trough time
    varies LESS across the RT groups is the one the dip is locked to.
    """
    tr = trials.dropna(subset=["rt"]).copy()
    tr["grp"] = pd.qcut(tr["rt"], n_groups, labels=False)
    rows = []
    for g, sub in tr.groupby("grp"):
        row = {"RT group": int(g) + 1, "n": len(sub), "median RT (s)": round(float(sub["rt"].median()), 3)}
        for name, bc, win in (("tone", bc_tone, search_tone), ("release", bc_release, search_release)):
            curve = np.convolve(bc.get(sub).mean(0), np.ones(smooth) / smooth, mode="same")
            m = (bc.times >= win[0]) & (bc.times <= win[1])
            row[f"trough rel. {name} (s)"] = round(float(bc.times[m][np.argmin(curve[m])]), 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    sp_t = df["trough rel. tone (s)"].max() - df["trough rel. tone (s)"].min()
    sp_r = df["trough rel. release (s)"].max() - df["trough rel. release (s)"].min()
    df.attrs["summary"] = (f"trough spread across RT groups: tone-locked {sp_t:.3f} s, release-locked "
                           f"{sp_r:.3f} s -> locked to {'RELEASE' if sp_r < sp_t else 'TONE'}")
    return df


def pre_tone_rt_corr(bc_tone, groups, keys, window=(-0.3, 0.0), n_perm=5000, seed=0):
    """
    Within a foreperiod: does pre-tone power predict the reaction time? Spearman rank correlation of
    mean power in `window` (relative to the tone) with RT, permutation p. Keys = group keys.
    """
    rng = np.random.default_rng(seed)
    m = (bc_tone.times >= window[0] - 1e-9) & (bc_tone.times <= window[1] + 1e-9)
    rows = []
    for key in keys:
        tr = groups[key].dropna(subset=["rt"])
        tr = tr[[t in bc_tone._pos for t in tr.index]]
        x, y = bc_tone.get(tr)[:, m].mean(1), tr["rt"].values
        rho = spearmanr(x, y)[0]
        null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
        rows.append({"group": group_label(key), "n": len(tr), "rho": round(float(rho), 3),
                     "p": round(float((1 + np.sum(np.abs(null) >= abs(rho))) / (n_perm + 1)), 4)})
    return pd.DataFrame(rows)


def plot_curves(bc, trial_sets, title, shade=None, save_path=None, show=False):
    """
    Mean band power (percent change) with +/- 1 SEM for several trial sets.
    trial_sets = {"correct (n=..)": DataFrame, ...}; shade = [(t0, t1), ...] windows to mark.
    """
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for label, tr in trial_sets.items():
        d = bc.get(tr)
        m, se = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
        ax.plot(bc.times, _pct(m), linewidth=1.8, label=f"{label} (n={len(d)})")
        ax.fill_between(bc.times, _pct(m - se), _pct(m + se), alpha=0.2, linewidth=0)
    for lo, hi in (shade or []):
        ax.axvspan(lo, hi, color="grey", alpha=0.12, linewidth=0)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Power change vs. baseline (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ======================================================================
# 8. BEHAVIOUR HELPERS (licks, reward timing)
def load_licks(session_dir, placeholder=-3.77):
    """Lick onsets (s, behaviour clock) from licks.npz; the placeholder value is dropped."""
    z = np.load(os.path.join(str(session_dir), "licks.npz"))
    on = np.asarray(z["onset_s"], float)
    return np.sort(on[~np.isclose(on, placeholder) & np.isfinite(on)])


def lick_tables(beh, licks, outcomes=("correct", "late", "early"), bin_s=0.1, span=(-0.5, 1.0)):
    """
    Licks relative to the release (behaviour clock). Returns (per-bin licks per trial,
    first-lick-after-release summary). The release column is on the same clock as the licks.
    """
    edges = np.arange(span[0], span[1] + 1e-9, bin_s)
    binned, first = {}, []
    for o in outcomes:
        rel = beh.loc[beh[o] == 1, "release"].dropna().values
        counts = np.zeros(len(edges) - 1)
        lat = []
        for r in rel:
            x = licks - r
            counts += np.histogram(x, edges)[0]
            after = x[(x >= 0) & (x <= span[1])]
            lat.append(after.min() if len(after) else np.nan)
        binned[o] = counts / max(len(rel), 1)
        lat = np.array(lat)
        ok = lat[np.isfinite(lat)]
        first.append({"outcome": o, "n_trials": len(rel), "% with a lick": round(100 * len(ok) / max(len(rel), 1)),
                      "latency q10": np.nan if not len(ok) else round(float(np.percentile(ok, 10)), 3),
                      "median": np.nan if not len(ok) else round(float(np.median(ok)), 3),
                      "q90": np.nan if not len(ok) else round(float(np.percentile(ok, 90)), 3)})
    tab = pd.DataFrame(binned, index=[f"{lo:+.1f}..{lo + bin_s:+.1f}" for lo in edges[:-1]]).round(2)
    tab.index.name = "bin (s rel. release)"
    return tab, pd.DataFrame(first)


def reward_delay(beh):
    """Reward delivery minus release (s) for the rewarded trials: shows how tightly they are locked."""
    d = (beh["reward_delivery"] - beh["release"])[beh["reward_delivery"].notna() & (beh["correct"] == 1)]
    return d.describe().round(3)


# ======================================================================
# 9. SIGN-FLIP PERMUTATION TEST (trial-level, per channel per band)
def sign_flip_test(values, n_perm=5000, seed=0):
    """
    Sign-flip permutation test on trial-level values. Tests H0: mean = 0.
    Each permutation randomly flips the sign of each value, then computes the mean.
    p = fraction of null means >= |observed mean|.

    Parameters
    ----------
    values : 1D array of % change values (one per trial)
    n_perm : number of permutations
    seed : random seed

    Returns
    -------
    dict with keys: mean, p, n, significant (p < 0.05), direction ("UP" or "DOWN")
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return {"mean": float(np.mean(values)), "p": 1.0, "n": n,
                "significant": False, "direction": "NS"}

    observed_mean = np.mean(values)
    rng = np.random.default_rng(seed)

    # sign-flip: multiply each value by +1 or -1 randomly
    signs = rng.choice([-1, 1], size=(n_perm, n))
    null_means = (signs * values[np.newaxis, :]).mean(axis=1)

    p = (1 + np.sum(np.abs(null_means) >= np.abs(observed_mean))) / (n_perm + 1)

    return {"mean": float(observed_mean), "p": float(p), "n": n,
            "significant": p < 0.05,
            "direction": "UP" if observed_mean > 0 else "DOWN"}


def run_sign_flip(ds_data, cond_trials, all_trials, channel_ids, band_settings, band_ranges,
                  event_col, baseline_window, hold_window, fs, tmin, tmax,
                  n_perm=5000, seed=0, decim=10, band_specific=None):
    """
    Sign-flip permutation test across all channels and bands.

    For each channel:
      1. Extract single-trial power for ALL trials (for the pooled baseline).
      2. Extract single-trial power for the CONDITION trials aligned to event_col.
      3. Compute log10(power / baseline) per trial.
      4. Average over time in hold_window and over frequencies in each band -> one value per trial.
      5. Convert to % change: 100 * (10^value - 1).
      6. Run sign_flip_test on those values.

    Parameters
    ----------
    ds_data : recording object
    cond_trials : DataFrame of the condition trials to test
    all_trials : DataFrame of ALL trials (for pooled baseline)
    channel_ids : list of channel IDs to test
    band_settings : list of dicts with 'freqs' and 'n_cycles'
    band_ranges : dict like {"theta": (4, 8), ...}
    event_col : str, the event column for alignment (e.g. "press_np_s")
    baseline_window : tuple (start, end) in seconds
    hold_window : tuple (start, end) in seconds — the window to average over
    fs : sampling rate
    tmin, tmax : extraction window
    n_perm : number of permutations
    seed : random seed
    decim : decimation factor
    band_specific : None (test all bands) or str like "theta" (test only that band)

    Returns
    -------
    DataFrame with columns: channel, band, n_trials, mean_log, mean_pct, p, significant, direction
    """
    # which bands to test?
    if band_specific is not None:
        test_bands = {band_specific: band_ranges[band_specific]}
    else:
        test_bands = band_ranges

    freqs_all = np.concatenate([b["freqs"] for b in band_settings])
    press_col = "press_np_s"  # baseline is always from press

    rows = []
    for ch in channel_ids:
        print(f"  sign-flip: {ch}", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            # baseline from all trials aligned to press
            pw_base, t_base, _, _ = extract_single_trial_power(
                ds_data, all_trials.dropna(subset=[press_col]), ch, press_col,
                band_settings, tmin, tmax, fs,
                keep_window=(baseline_window[0] - 0.1, baseline_window[1] + 0.1), decim=decim)
        base = compute_pooled_baseline(pw_base, t_base, baseline_window)

        with contextlib.redirect_stdout(io.StringIO()):
            # condition trials aligned to event_col
            pw_cond, t_cond, _, kept = extract_single_trial_power(
                ds_data, cond_trials.dropna(subset=[event_col]), ch, event_col,
                band_settings, tmin, tmax, fs,
                keep_window=(hold_window[0] - 0.1, hold_window[1] + 0.1), decim=decim)

        # log baseline
        log_power = apply_log_baseline(pw_cond, base)  # (trials, freqs, times)

        # time mask for the hold window
        t_mask = (t_cond >= hold_window[0] - 1e-9) & (t_cond <= hold_window[1] + 1e-9)

        for band_name, (flo, fhi) in test_bands.items():
            f_mask = (freqs_all >= flo) & (freqs_all <= fhi)
            # mean over frequencies and time -> one value per trial (in log10 units)
            trial_vals_log = log_power[:, f_mask, :][:, :, t_mask].mean(axis=(1, 2))
            # convert to % change
            trial_vals_pct = 100 * (10 ** trial_vals_log - 1)

            result = sign_flip_test(trial_vals_pct, n_perm=n_perm, seed=seed)

            rows.append({
                "channel": ch,
                "band": band_name,
                "n_trials": result["n"],
                "mean_log": round(float(np.mean(trial_vals_log)), 4),
                "mean_pct": round(result["mean"], 2),
                "p": round(result["p"], 4),
                "significant": result["significant"],
                "direction": result["direction"],
            })

    return pd.DataFrame(rows)

# ======================================================================
# 10. TWO-GROUP CLUSTER TEST ON MANY CHANNELS  (A vs B; test on log10, plots in %)
import pickle
from matplotlib.patches import Polygon


def _band_hits(mask, freqs, bands):
    """Names of the bands that a cluster mask (freqs x times) touches."""
    f = np.asarray(freqs)[mask.any(axis=1)]
    return [b for b, (lo, hi) in bands.items() if ((f >= lo) & (f <= hi)).any()]


def _cache_file(out_dir, ref, ch):
    return os.path.join(out_dir, ref, "_channels", slugify(ch) + ".pkl")


def run_two_group(recordings, groups, all_trials, key_a, key_b, labels, channels, CONFIG, fs,
                  tests, stat_bands, out_dir, n_perm=5000, pval=0.05, seed=0, decim=10, min_n=10):
    """
    Cluster permutation test A vs B for every channel, every reference and every alignment.

    tests  {"Press":   {"window": (-0.3, 0.3),  "allow_pre": True},
            "Release": {"window": (-0.05, 0.5), "cross_release": True}}
    labels ("correct", "early")  -> direction names "correct > early" / "early > correct"

    Per channel:
      1. single-trial Morlet power, Press-aligned, ALL trials -> pooled baseline (CONFIG baseline_window)
      2. per alignment: eligible trials of A and B (whole test window inside the trial)
      3. test on log10(power / baseline) inside the test window, whole frequency map
      4. % maps for the plots: (mean power / baseline - 1) * 100 over the plot window
    Each channel is saved at once to out_dir/<ref>/_channels/<channel>.pkl.
    A channel whose file exists is skipped -> a stopped run continues where it stopped.
    """
    cols, bw = CONFIG["events"], CONFIG["baseline_window"]
    press_keep = (min(PLOT_WINDOWS["Press"][0], bw[0]), max(PLOT_WINDOWS["Press"][1], bw[1]))
    ab = pd.concat([groups[key_a], groups[key_b]])
    ab = ab[~ab.index.duplicated()]
    dir_pos, dir_neg = f"{labels[0]} > {labels[1]}", f"{labels[1]} > {labels[0]}"

    for ref, ds in recordings.items():
        os.makedirs(os.path.join(out_dir, ref, "_channels"), exist_ok=True)
        for i, ch in enumerate(channels, 1):
            f_out = _cache_file(out_dir, ref, ch)
            if os.path.exists(f_out):
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                p_pow, p_times, freqs, p_kept = extract_single_trial_power(
                    ds, all_trials, ch, cols["Press"], CONFIG["band_settings"],
                    CONFIG["tmin"], CONFIG["tmax"], fs, keep_window=press_keep, decim=decim)
            base = compute_pooled_baseline(p_pow, p_times, bw)
            res, msg = {}, []
            for al, spec in tests.items():
                if al == "Press":
                    power, times, kept = p_pow, p_times, p_kept
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        power, times, _, kept = extract_single_trial_power(
                            ds, ab, ch, cols[al], CONFIG["band_settings"], CONFIG["tmin"],
                            CONFIG["tmax"], fs, keep_window=PLOT_WINDOWS[al], decim=decim)
                win = spec["window"]
                m = (times >= win[0] - 1e-9) & (times <= win[1] + 1e-9)
                pos = {t: k for k, t in enumerate(kept)}
                rows = []
                for key in (key_a, key_b):
                    tr = eligible_trials(groups[key], al, win,
                                         cross_release=spec.get("cross_release", False),
                                         allow_pre=spec.get("allow_pre", False))
                    rows.append([pos[t] for t in tr.index if t in pos])
                nA, nB = len(rows[0]), len(rows[1])
                if min(nA, nB) < min_n:
                    res[al] = {"tested": False, "nA": nA, "nB": nB}
                    msg.append(f"{al}: skipped (nA={nA}, nB={nB})")
                    continue
                d = [apply_log_baseline(power[r][:, :, m], base) for r in rows]
                r = permutation_test(d[0], d[1], n_perm, pval, seed)
                t_test = times[m]
                clus = []
                for c in r["clusters"]:
                    ff, tt = np.where(c["mask"])
                    clus.append({"direction": dir_pos if c["sign"] > 0 else dir_neg,
                                 "cluster p": round(c["p"], 4), "significant": c["p"] < pval,
                                 "t-mass": round(c["mass"], 1), "size (pixels)": c["size"],
                                 "t_start (s)": round(float(t_test[tt.min()]), 3),
                                 "t_end (s)": round(float(t_test[tt.max()]), 3),
                                 "f_min (Hz)": float(freqs[ff.min()]), "f_max (Hz)": float(freqs[ff.max()]),
                                 "bands": "; ".join(_band_hits(c["mask"], freqs, stat_bands))})
                res[al] = {"tested": True, "nA": nA, "nB": nB, "p_min": r["p_min"],
                           "A": ((power[rows[0]].mean(0) / base - 1) * 100).astype(np.float32),
                           "B": ((power[rows[1]].mean(0) / base - 1) * 100).astype(np.float32),
                           "times": times, "freqs": freqs, "sig": r["sig_corrected"], "sig_t": t_test,
                           "clusters": clus}
                msg.append(f"{al}: nA={nA} nB={nB} p_min={r['p_min']:.4f}")
            with open(f_out, "wb") as f:
                pickle.dump(res, f)
            print(f"[{ref}] {i}/{len(channels)} {ch} | " + " | ".join(msg), flush=True)


def load_two_group(out_dir, ref, channels):
    """{channel: result dict} from the per-channel files (missing channels are left out)."""
    out = {}
    for ch in channels:
        f = _cache_file(out_dir, ref, ch)
        if os.path.exists(f):
            with open(f, "rb") as fh:
                out[ch] = pickle.load(fh)
    return out


def two_group_tables(out_dir, refs, alignments, probe_df, labels, stat_bands):
    """
    Writes  out_dir/<ref>/<alignment>/results.xlsx  (sheets: channels, clusters)
            out_dir/summary.xlsx                     (one sheet per alignment)
    A channel counts for a band + direction if a significant cluster (p < pval) of that
    direction touches the band. Returns {alignment: summary DataFrame}.
    """
    dirs = (f"{labels[0]} > {labels[1]}", f"{labels[1]} > {labels[0]}")
    ys = probe_df.set_index("channel_id")["y"]
    summaries = {}
    for al in alignments:
        summ = {}
        for ref in refs:
            res = load_two_group(out_dir, ref, probe_df["channel_id"])
            ch_rows, cl_rows = [], []
            for ch, rr in res.items():
                r = rr.get(al)
                if r is None:
                    continue
                row = {"channel": ch, "y (um)": float(ys.get(ch, np.nan)), "tested": r["tested"],
                       "nA": r["nA"], "nB": r["nB"]}
                if r["tested"]:
                    sig = [c for c in r["clusters"] if c["significant"]]
                    row.update({"p_min": round(r["p_min"], 4), "n significant clusters": len(sig),
                                "biggest significant t-mass": max([c["t-mass"] for c in sig], default=0.0)})
                    for b in stat_bands:
                        for dname in dirs:
                            row[f"{b}: {dname}"] = any(c["direction"] == dname and b in c["bands"].split("; ")
                                                       for c in sig)
                    cl_rows += [{"channel": ch, "y (um)": row["y (um)"], **c} for c in sig]
                ch_rows.append(row)
            chs = pd.DataFrame(ch_rows).sort_values("y (um)").reset_index(drop=True)
            cls = pd.DataFrame(cl_rows)
            folder = os.path.join(out_dir, ref, al.lower())
            os.makedirs(folder, exist_ok=True)
            with pd.ExcelWriter(os.path.join(folder, "results.xlsx")) as xw:
                chs.to_excel(xw, sheet_name="channels", index=False)
                (cls if len(cls) else pd.DataFrame([{"channel": "no significant cluster"}])
                 ).to_excel(xw, sheet_name="clusters", index=False)
            tested = chs[chs["tested"]] if len(chs) else chs
            for dname in dirs:
                col = []
                for b, (lo, hi) in stat_bands.items():
                    col.append(int(tested[f"{b}: {dname}"].sum()) if len(tested) else 0)
                any_band = tested[[f"{b}: {dname}" for b in stat_bands]].any(axis=1).sum() if len(tested) else 0
                summ[f"{ref}: {dname}"] = col + [int(any_band), len(tested)]
        idx = [f"{b} ({lo}-{hi} Hz)" for b, (lo, hi) in stat_bands.items()] + \
              ["Any band (total significant channels)", "Channels tested"]
        summaries[al] = pd.DataFrame(summ, index=idx)
    with pd.ExcelWriter(os.path.join(out_dir, "summary.xlsx")) as xw:
        for al, df in summaries.items():
            df.to_excel(xw, sheet_name=al)
    return summaries


def plot_two_group_part(ch_res, alignment, part_df, probe_df, labels, title, freq_range, yticks,
                        band_boundaries=None, vmax=100, diff_vmax=None, contour=None,
                        save_path=None, dpi=200):
    """Rows = channels. Columns: A | B | A - B | probe. All in % change; white outline = significant cluster."""
    n = len(part_df)
    fig_h = 3.2 * n + 1.8
    fig = plt.figure(figsize=(19.5, fig_h))
    gs = fig.add_gridspec(n, 11, wspace=0.0, hspace=0.45,
                          width_ratios=[1, 0.10, 1, 0.06, 0.035, 0.34, 1, 0.06, 0.035, 0.42, 0.7],
                          left=0.06, right=0.97, top=1 - 1.4 / fig_h, bottom=0.6 / fig_h)
    im_ab = im_d = None
    for r, (_, row) in enumerate(part_df.iterrows()):
        ch = row["channel_id"]
        d = ch_res[ch][alignment]
        times, freqs = d["times"], d["freqs"]
        panels = [(d["A"], vmax, f"A: {labels[0]} (n={d['nA']})"),
                  (d["B"], vmax, f"B: {labels[1]} (n={d['nB']})"),
                  (d["A"] - d["B"], diff_vmax, "A - B")]
        for k, (arr, lim, name) in enumerate(panels):
            ax = fig.add_subplot(gs[r, [0, 2, 6][k]])
            im = ax.pcolormesh(times, freqs, arr, cmap="viridis", vmin=-lim, vmax=lim, shading="gouraud")
            _outline(ax, d["sig_t"], freqs, d["sig"])
            ax.axvline(0, color="white", linestyle="--", linewidth=1.2)
            for f in (band_boundaries or []):
                ax.axhline(f, color="white", linestyle=":", linewidth=0.6)
            ax.set_ylim(freq_range); ax.set_yticks(yticks); ax.set_xlim(times[0], times[-1])
            ax.grid(False); ax.tick_params(labelsize=7); ax.set_xlabel("Time (s)", fontsize=8)
            if k == 0:
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.text(-0.22, 0.5, ch, transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=13, fontweight="bold")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                ax.set_title(name.split(" (n=")[0] if k < 2 else name, fontsize=12, fontweight="bold", pad=8)
            ax.text(0.98, 0.95, f"n={d['nA']}" if k == 0 else (f"n={d['nB']}" if k == 1 else ""),
                    transform=ax.transAxes, ha="right", va="top", color="white", fontsize=8)
            if k < 2:
                im_ab = im
            else:
                im_d = im
        axp = fig.add_subplot(gs[r, 10])
        tfx.draw_probe(axp, probe_df, row["x"], row["y"], contour)
        if r == 0:
            axp.set_title("Probe location", fontsize=12, fontweight="bold", pad=8)
    cb1 = fig.colorbar(im_ab, cax=fig.add_subplot(gs[:, 4]), extend="both")
    t1 = np.linspace(-vmax, vmax, 9)
    cb1.set_ticks(t1); cb1.set_ticklabels([f"{t:+.0f}%" if t else "0%" for t in t1])
    cb1.set_label("% change from baseline", fontsize=9, labelpad=8)
    cb1.ax.tick_params(labelsize=7)
    cb2 = fig.colorbar(im_d, cax=fig.add_subplot(gs[:, 8]), extend="both")
    t2 = np.linspace(-diff_vmax, diff_vmax, 5)
    cb2.set_ticks(t2); cb2.set_ticklabels([f"{t:+.0f}" if abs(t) > 1e-9 else "0" for t in t2])
    cb2.set_label("A - B (percentage points): yellow = A higher, purple = B higher", fontsize=9, labelpad=8)
    cb2.ax.tick_params(labelsize=7)
    fig.suptitle(title + "\nwhite outline = significant cluster (raw cluster p < 0.05)",
                 fontsize=15, fontweight="bold", y=1 - 0.2 / fig_h, va="top")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_probe_bands(channels_df, probe_df, labels, stat_bands, band_colors, title,
                     contour=None, save_path=None, dpi=200):
    """
    One probe per band. Grey = all contacts. Significant channels in the band colour:
    empty circle = A > B, filled circle = B > A, filled circle with thick black ring = both.
    channels_df = the 'channels' table of two_group_tables (columns '<band>: <direction>').
    """
    dpos, dneg = f"{labels[0]} > {labels[1]}", f"{labels[1]} > {labels[0]}"
    xy = probe_df.set_index("channel_id")[["x", "y"]]
    fig, axes = plt.subplots(1, len(stat_bands), figsize=(3.2 * len(stat_bands), 11))
    for ax, (b, (lo, hi)) in zip(axes, stat_bands.items()):
        col = band_colors[b]
        if contour is not None:
            ax.add_patch(Polygon(contour, closed=True, facecolor="#f2f2f2", edgecolor="#888888",
                                 linewidth=1.0, zorder=0))
        ax.scatter(probe_df["x"], probe_df["y"], s=5, marker="s", color="#c8c8c8", edgecolors="none", zorder=1)
        t = channels_df[channels_df["tested"]] if len(channels_df) else channels_df
        pos = t[f"{b}: {dpos}"].astype(bool) if len(t) else pd.Series(dtype=bool)
        neg = t[f"{b}: {dneg}"].astype(bool) if len(t) else pd.Series(dtype=bool)
        only_pos = t.loc[pos & ~neg, "channel"] if len(t) else []
        only_neg = t.loc[neg & ~pos, "channel"] if len(t) else []
        both = t.loc[pos & neg, "channel"] if len(t) else []
        if len(only_pos):
            p = xy.loc[list(only_pos)]
            ax.scatter(p["x"], p["y"], s=60, facecolors="none", edgecolors=col, linewidths=1.8, zorder=3)
        if len(only_neg):
            p = xy.loc[list(only_neg)]
            ax.scatter(p["x"], p["y"], s=60, color=col, edgecolors=col, linewidths=1.0, zorder=3)
        if len(both):
            p = xy.loc[list(both)]
            ax.scatter(p["x"], p["y"], s=80, color=col, edgecolors="black", linewidths=2.5, zorder=4)
        ax.set_title(f"{b} ({lo}-{hi} Hz)\n{len(only_pos)} empty, {len(only_neg)} filled, {len(both)} both",
                     fontsize=11, fontweight="bold", color=col)
        ax.set_xlim(probe_df["x"].min() - 150, probe_df["x"].max() + 150)
        ax.set_ylim(probe_df["y"].min() - 150, probe_df["y"].max() + 150)
        ax.set_xticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Depth along probe, y (um)  (0 = tip)", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor="none",
                          markeredgecolor="dimgray", markeredgewidth=1.8, label=dpos),
               plt.Line2D([], [], marker="o", ls="", markersize=9, color="dimgray", label=dneg),
               plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor="dimgray",
                          markeredgecolor="black", markeredgewidth=2.5, label="both directions")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle(title + "\nsignificant = raw cluster p < 0.05, cluster touches the band",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
