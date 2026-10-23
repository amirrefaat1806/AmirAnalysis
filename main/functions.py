"""
functions.py  -  shared functions for the main analysis (all final sessions).
Sections
  1. Loading      load_session, probe_table
  2. Signal QC    compute_psd, psd_table, flag_channels_local
  3. QC plots     plot_psd_depth, plot_mean_psd, plot_traces
"""
import os
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════════
# 1. LOADING
# ════════════════════════════════════════════════════════════════════════
def load_session(proc, session, extra_bad=None, refs=("original", "CMR"),
                 sub=os.path.join("amir_data", "down_sampled", "1000_Hz")):
    """Load one session's un-referenced 1000 Hz recording, remove its extra bad channels, build the references.

    proc      : path of 2_processed
    extra_bad : table with columns session, channel_id (extra_bad_channels.csv) or None
    refs      : which versions to return, any of "original", "CMR", "LOCAL"
                CMR   = global median over all remaining channels
                LOCAL = median of the channels 30-200 µm away (SpikeInterface local reference, as in band_power_v2_localref)
    returns   : dict {ref: recording}, list of removed channel ids
    The channels removed in preprocessing are already missing from the saved recording; extra bad channels are
    removed BEFORE referencing, so they never enter a median. References are applied on the fly (nothing is saved).
    """
    import re
    import spikeinterface as si
    import spikeinterface.preprocessing as spre
    load = getattr(si, "load", None) or si.load_extractor               # si.load (0.105) / si.load_extractor (older)
    rec = load(os.path.join(proc, session, sub))

    removed = []
    if extra_bad is not None:
        want = set(extra_bad.loc[extra_bad["session"] == session, "channel_id"].astype(str)
                   .str.extract(r"(AP\d+)")[0].dropna())                # "AP362" or "imec0.ap#AP362" -> "AP362"
        ids = [str(c) for c in rec.get_channel_ids()]
        removed = [c for c in ids if re.search(r"AP\d+", c).group(0) in want]
        if removed:
            rec = rec.remove_channels(removed)

    out = {}
    for r in refs:
        if r == "original":
            out[r] = rec
        elif r == "CMR":
            out[r] = spre.common_reference(rec, reference="global", operator="median")
        elif r == "LOCAL":
            out[r] = spre.common_reference(rec, reference="local", operator="median", local_radius=(30, 200))
        else:
            raise ValueError(f"unknown reference {r!r}")
    return out, removed


def probe_table(rec):
    """One row per recorded channel: channel_id, x, y (µm, y = distance from the tip), shank (NP2.0: 250 µm apart)."""
    loc = rec.get_channel_locations()
    df = pd.DataFrame({"channel_id": [str(c) for c in rec.get_channel_ids()], "x": loc[:, 0], "y": loc[:, 1]})
    df["shank"] = (df["x"] // 250).astype(int)
    return df

# ════════════════════════════════════════════════════════════════════════
# 2. SIGNAL QC
# ════════════════════════════════════════════════════════════════════════
QC_BANDS = {"delta": (1, 3), "theta": (4, 7), "alpha": (8, 12), "beta": (13, 29),
            "low_gamma": (30, 60), "high_gamma": (61, 100)}


def compute_psd(rec, channel_ids=None, n_segments=30, seg_s=10.0, fmax=100.0, nperseg_s=2.0):
    """
    Welch PSD per channel from `n_segments` pieces of `seg_s` seconds spread evenly over the whole recording
    (reading the whole recording at once would need several GB of RAM).
    Welch window nperseg_s = 2 s -> 0.5 Hz resolution.
    Returns (freqs, psd, channel_ids): psd = (n_channels, n_freqs) in µV²/Hz, mean over segments, 1..fmax Hz.
    """
    from scipy.signal import welch
    fs = rec.get_sampling_frequency()
    channel_ids = list(rec.get_channel_ids() if channel_ids is None else channel_ids)
    n_total = rec.get_num_samples()
    n_seg = int(seg_s * fs)
    starts = np.linspace(0, n_total - n_seg, n_segments).astype(int)
    acc = None
    for s in starts:
        x = rec.get_traces(start_frame=int(s), end_frame=int(s) + n_seg,
                           channel_ids=channel_ids, return_in_uV=True).astype(np.float64)
        f, p = welch(x, fs=fs, nperseg=int(nperseg_s * fs), axis=0)       # p: (n_freqs, n_channels)
        acc = p if acc is None else acc + p
    psd = (acc / len(starts)).T
    keep = (f >= 1) & (f <= fmax)
    return f[keep], psd[:, keep], [str(c) for c in channel_ids]


def psd_table(freqs, psd, channel_ids, probe_df, bands=None, line_hz=50.0):
    """
    One row per channel: shank, x, y, log10 power per band, broadband (1-100 Hz) log power,
    deviation from the shank median in MAD units (information only; single bad channels: flag_channels_local),
    and the 50 Hz line ratio = power 49-51 Hz / power 45-48 + 52-55 Hz (~1 or below = notch worked).
    """
    bands = bands or QC_BANDS
    df = pd.DataFrame({"channel_id": [str(c) for c in channel_ids]})
    df = df.merge(probe_df[["channel_id", "shank", "x", "y"]], on="channel_id", how="left")
    for b, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs <= hi)
        df[f"log_{b}"] = np.log10(psd[:, m].mean(axis=1))
    df["log_broadband"] = np.log10(psd.mean(axis=1))
    med = df.groupby("shank")["log_broadband"].transform("median")
    mad = df.groupby("shank")["log_broadband"].transform(lambda v: np.median(np.abs(v - np.median(v))))
    df["broadband_dev_mad"] = (df["log_broadband"] - med) / mad.replace(0, np.nan)
    on = (freqs >= line_hz - 1) & (freqs <= line_hz + 1)
    off = ((freqs >= line_hz - 5) & (freqs <= line_hz - 2)) | ((freqs >= line_hz + 2) & (freqs <= line_hz + 5))
    df["line_ratio"] = psd[:, on].mean(axis=1) / psd[:, off].mean(axis=1)
    return df

def flag_channels_local(table, col="log_broadband", radius_um=75.0, thr_log=0.3, only_flagged=True):
    """
    Single-channel outliers: each channel vs the median of its neighbours on the same shank
    (|y difference| <= radius_um, the channel itself excluded; needs >= 3 neighbours).
    thr_log = 0.3 log10 units = factor 2. Depth blocks (several neighbouring channels together) are NOT flagged.
    Adds 'local_dev' (log10 difference to the neighbour median) and 'flag_local'.
    only_flagged=True returns the flagged rows only (sorted by shank, y); False returns all rows.
    """
    t = table.copy()
    dev = np.full(len(t), np.nan)
    for sh, g in t.groupby("shank"):
        y, v = g["y"].values, g[col].values
        for k, i in enumerate(g.index):
            nb = (np.abs(y - y[k]) <= radius_um) & (np.arange(len(y)) != k)
            if nb.sum() >= 3:
                dev[t.index.get_loc(i)] = v[k] - np.median(v[nb])
    t["local_dev"] = dev
    t["flag_local"] = np.abs(t["local_dev"]) > thr_log
    return t[t["flag_local"]].sort_values(["shank", "y"]) if only_flagged else t

def qc_session(recs, probe_df, n_segments=30, seg_s=10.0, radius_um=75.0, thr_log=0.3):
    """
    Signal QC of one session for every reference in recs ({"original": rec, "CMR": rec}).
    Returns freqs, {ref: psd (channels, freqs)}, table (one row per channel x ref) with band powers,
    line_ratio (50 Hz), local_dev and flag_local (fn.flag_channels_local, all channels kept).
    """
    psd, tabs, freqs = {}, [], None
    for ref, rec in recs.items():
        freqs, psd[ref], ids = compute_psd(rec, n_segments=n_segments, seg_s=seg_s)
        t = psd_table(freqs, psd[ref], ids, probe_df)
        t = flag_channels_local(t, radius_um=radius_um, thr_log=thr_log, only_flagged=False)
        t.insert(0, "ref", ref)
        tabs.append(t)
    return freqs, psd, pd.concat(tabs, ignore_index=True)

# ════════════════════════════════════════════════════════════════════════
# 3. QC PLOTS
# ════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt


def select_channels(probe_df, n_total=20, min_per_shank=2):
    """n_total channels spread over the shanks in proportion to their channel counts (>= min_per_shank each),
    evenly over depth within each shank (as in the pilot, section 2)."""
    counts = probe_df["shank"].value_counts().sort_index()
    n_per = (counts / counts.sum() * n_total).round().astype(int).clip(lower=min_per_shank)
    sel = []
    for sh, d in probe_df.sort_values("y").groupby("shank"):
        idx = np.linspace(0, len(d) - 1, n_per[sh]).round().astype(int)
        sel += d.iloc[idx]["channel_id"].tolist()
    return probe_df[probe_df["channel_id"].isin(sel)].reset_index(drop=True)


def _depth_grid(L, y, gap_um=30.0, step_um=15.0):
    """Average channels at the same depth (2 columns) and insert empty (NaN) rows where the map has
    a gap, so the y axis is true to scale. Returns (grid, y_edges)."""
    uy = np.unique(y)
    rows = np.array([L[y == v].mean(axis=0) for v in uy])
    out_rows, out_y = [], []
    for k, v in enumerate(uy):
        if k and uy[k] - uy[k - 1] > gap_um:                      # gap: fill with NaN rows
            for g in np.arange(uy[k - 1] + step_um, v, step_um):
                out_rows.append(np.full(rows.shape[1], np.nan)); out_y.append(g)
        out_rows.append(rows[k]); out_y.append(v)
    out_y = np.array(out_y)
    edges = np.concatenate([[out_y[0] - step_um / 2], (out_y[:-1] + out_y[1:]) / 2, [out_y[-1] + step_um / 2]])
    return np.array(out_rows), edges


def _save_show(fig, save_path, show):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_psd_depth(freqs, psd, channel_ids, probe_df, title, save_path=None, show=True):
    """Per shank, two panels: (top) log10 PSD, depth x frequency; (bottom) same minus the shank's median spectrum,
    so single odd channels stand out. y axis true to scale (gaps white); deepest (small y) at the bottom."""
    df = pd.DataFrame({"channel_id": channel_ids, "row": np.arange(len(channel_ids))})
    df = df.merge(probe_df[["channel_id", "shank", "y"]], on="channel_id", how="left")
    shanks = sorted(df["shank"].dropna().unique())
    logp = np.log10(psd)
    f_edges = np.concatenate([[freqs[0] - (freqs[1] - freqs[0]) / 2], (freqs[:-1] + freqs[1:]) / 2,
                              [freqs[-1] + (freqs[-1] - freqs[-2]) / 2]])
    fig, axes = plt.subplots(2, len(shanks), figsize=(4.5 * len(shanks), 10), squeeze=False)
    v_abs = np.nanpercentile(logp, [2, 98])
    for j, sh in enumerate(shanks):
        d = df[df["shank"] == sh]
        L = logp[d["row"].values]
        rel = L - np.median(L, axis=0, keepdims=True)
        G, y_edges = _depth_grid(L, d["y"].values)
        R, _ = _depth_grid(rel, d["y"].values)
        ax = axes[0, j]
        im0 = ax.pcolormesh(f_edges, y_edges, G, cmap="viridis", vmin=v_abs[0], vmax=v_abs[1], shading="flat")
        ax.set_title(f"shank {int(sh)} ({len(d)} ch)\nlog10 PSD", fontsize=10, fontweight="bold")
        ax = axes[1, j]
        im1 = ax.pcolormesh(f_edges, y_edges, R, cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="flat")
        ax.set_title("minus shank median", fontsize=10)
        for a in axes[:, j]:
            a.set_xlabel("Frequency (Hz)", fontsize=8)
            a.axvline(50, color="white" if a is axes[0, j] else "k", lw=0.6, ls=":")
            a.tick_params(labelsize=7)
            a.set_ylabel("y (µm)", fontsize=8)
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01, label="log10 PSD (µV²/Hz)")
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01, label="Δ log10 vs shank median")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _save_show(fig, save_path, show)


def plot_mean_psd(psd_by_ref, freqs, title, save_path=None, show=True):
    """Median spectrum over channels (with 25-75 % band) for each reference, e.g. original vs CMR."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, p in psd_by_ref.items():
        med = np.median(p, axis=0)
        q1, q3 = np.percentile(p, [25, 75], axis=0)
        ax.plot(freqs, med, label=name, lw=1.5)
        ax.fill_between(freqs, q1, q3, alpha=0.2)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 13, 30, 50, 100]); ax.set_xticklabels([1, 2, 4, 8, 13, 30, 50, 100])
    ax.axvline(50, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(title, fontweight="bold"); ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    _save_show(fig, save_path, show)


def plot_traces(rec, selected_df, t0_s, dur_s, title, spacing_uV=None, save_path=None, show=True):
    """A few seconds of raw LFP from the selected channels, deepest at the bottom."""
    fs = rec.get_sampling_frequency()
    d = selected_df.sort_values(["shank", "y"]).reset_index(drop=True)
    x = rec.get_traces(start_frame=int(t0_s * fs), end_frame=int((t0_s + dur_s) * fs),
                       channel_ids=d["channel_id"].tolist(), return_in_uV=True).astype(float)
    x = x - x.mean(axis=0, keepdims=True)
    spacing_uV = spacing_uV or 4 * np.nanmedian(np.std(x, axis=0))
    t = t0_s + np.arange(x.shape[0]) / fs
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(d) + 2))
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i] + i * spacing_uV, lw=0.5, color=f"C{int(d.loc[i, 'shank'])}")
    ax.set_yticks(np.arange(len(d)) * spacing_uV)
    ax.set_yticklabels([f"sh{int(r.shank)} y={int(r.y)}  {r.channel_id.split('AP')[-1]}"
                        for r in d.itertuples()], fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{title}  (spacing {spacing_uV:.0f} µV)", fontweight="bold")
    _save_show(fig, save_path, show)


# ════════════════════════════════════════════════════════════════════════
# 4. EVENT CHECK (event-locked averages)
# ════════════════════════════════════════════════════════════════════════
def event_erp(rec, event_times_s, channel_ids=None, tmin=-1.0, tmax=1.0):
    """
    Event-locked average (ERP) for every channel, without keeping all trials in memory.
    event_times_s: times in s on the recording clock (NaN and windows off the recording are skipped).
    Each trial: the mean of its pre-event part is subtracted (per channel).
    Returns dict: mean (n_times, n_ch), sem (n_times, n_ch), times, n (trials used), channel_ids.
    """
    fs = rec.get_sampling_frequency()
    channel_ids = [str(c) for c in (rec.get_channel_ids() if channel_ids is None else channel_ids)]
    n_pre, n_post = int(round(-tmin * fs)), int(round(tmax * fs))
    n_tot, n_samp = n_pre + n_post, rec.get_num_samples()
    times = (np.arange(n_tot) - n_pre) / fs
    s1 = s2 = None
    n = 0
    for t in np.asarray(event_times_s, float):
        if not np.isfinite(t):
            continue
        i0 = int(round(t * fs)) - n_pre
        if i0 < 0 or i0 + n_tot > n_samp:
            continue
        x = rec.get_traces(start_frame=i0, end_frame=i0 + n_tot, channel_ids=channel_ids,
                           return_in_uV=True).astype(np.float64)
        x -= x[:n_pre].mean(axis=0, keepdims=True)                 # remove the pre-event mean (per trial)
        s1 = x if s1 is None else s1 + x
        s2 = x ** 2 if s2 is None else s2 + x ** 2
        n += 1
    if n == 0:                                                       # no usable event: empty result instead of an error
        nan = np.full((n_tot, len(channel_ids)), np.nan)
        return {"mean": nan, "sem": nan.copy(), "times": times, "n": 0, "channel_ids": channel_ids}
    mean = s1 / n
    sem = np.sqrt(np.maximum(s2 / n - mean ** 2, 0) / max(n - 1, 1))
    return {"mean": mean, "sem": sem, "times": times, "n": n, "channel_ids": channel_ids}

def plot_erp_depth(erp, probe_df, title, vmax=None, marks=None, save_path=None, show=True):
    """
    Per shank: channels (true-scale depth, deepest at the bottom) x time, colour = ERP (µV).
    Top row: median over all channels (the part shared by the whole probe) and the shank mean.
    marks = {label: time_s} extra dotted lines (e.g. tone delay).
    """
    df = pd.DataFrame({"channel_id": erp["channel_ids"], "row": np.arange(len(erp["channel_ids"]))})
    df = df.merge(probe_df[["channel_id", "shank", "y"]], on="channel_id", how="left")
    shanks = sorted(df["shank"].dropna().unique())
    t = erp["times"]
    t_edges = np.concatenate([[t[0]], (t[:-1] + t[1:]) / 2, [t[-1]]])
    M = erp["mean"]
    vmax = vmax or np.nanpercentile(np.abs(M), 98)
    fig, axes = plt.subplots(2, len(shanks), figsize=(4.5 * len(shanks), 9), squeeze=False,
                             gridspec_kw={"height_ratios": [1, 3]})
    glob = np.median(M, axis=1)
    for j, sh in enumerate(shanks):
        d = df[df["shank"] == sh]
        ax = axes[0, j]
        ax.plot(t, glob, color="k", lw=1)
        ax.plot(t, M[:, d["row"].values].mean(axis=1), color=f"C{int(sh)}", lw=1)
        ax.axvline(0, color="r", lw=0.8, ls="--")
        ax.set_title(f"shank {int(sh)} ({len(d)} ch)\nblack = median all ch, colour = mean shank",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("µV", fontsize=8); ax.tick_params(labelsize=7)
        G, y_edges = _depth_grid(M[:, d["row"].values].T, d["y"].values)
        ax = axes[1, j]
        im = ax.pcolormesh(t_edges, y_edges, G, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        for lab, tm in (marks or {}).items():
            for a in axes[:, j]:
                a.axvline(tm, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("Time from event (s)", fontsize=8); ax.set_ylabel("y (µm)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01, label="ERP (µV)")
    fig.suptitle(f"{title}  (n = {erp['n']} trials)", fontsize=12, fontweight="bold")
    _save_show(fig, save_path, show)


def plot_erp_traces(erps, selected_df, title, save_path=None, show=True):
    """
    ERP (mean ± sem) of the selected channels, one small panel per channel (deepest at the bottom),
    several alignments / references overlaid: erps = {label: erp dict}.
    """
    d = selected_df.sort_values(["shank", "y"], ascending=[True, False]).reset_index(drop=True)
    n = len(d)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.9 * nrow), squeeze=False, sharex=True)
    for k, r in d.iterrows():
        ax = axes[k % nrow, k // nrow]
        for c, (lab, e) in enumerate(erps.items()):
            i = e["channel_ids"].index(r["channel_id"])
            m, s = e["mean"][:, i], e["sem"][:, i]
            ax.plot(e["times"], m, lw=1, color=f"C{c}", label=f"{lab} (n={e['n']})")
            ax.fill_between(e["times"], m - s, m + s, color=f"C{c}", alpha=0.2, lw=0)
        ax.axvline(0, color="r", lw=0.6, ls="--")
        ax.set_title(f"sh{int(r['shank'])} y={int(r['y'])} {r['channel_id'].split('AP')[-1]}", fontsize=7)
        ax.tick_params(labelsize=6)
    for k in range(n, nrow * ncol):
        axes[k % nrow, k // nrow].axis("off")
    axes[0, 0].legend(fontsize=6, loc="upper left")
    for a in axes[-1, :]:
        a.set_xlabel("Time from event (s)", fontsize=7)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_show(fig, save_path, show)

def alignment_sharpness(erp_a, erp_b, window=(-0.1, 0.3)):
    """
    Peak-to-peak of the ERP inside `window` per channel for two alignments of the same trials
    (e.g. press from TTL vs press from MA). A better alignment gives a larger, sharper average.
    Returns a table with ptp_a, ptp_b and ratio a/b per channel.
    """
    m = (erp_a["times"] >= window[0]) & (erp_a["times"] <= window[1])
    pa = np.ptp(erp_a["mean"][m], axis=0)
    pb = np.ptp(erp_b["mean"][m], axis=0)
    return pd.DataFrame({"channel_id": erp_a["channel_ids"], "ptp_a": pa, "ptp_b": pb, "ratio_a_b": pa / pb})


def erp_outliers(erp, probe_df, radius_um=75.0, thr_log=0.3):
    """Channels whose ERP size (SD of the average over time, log10) differs > thr_log from the median of their
    neighbours on the same shank (fn.flag_channels_local). Returns all channels with local_dev and flag_local."""
    t = pd.DataFrame({"channel_id": erp["channel_ids"], "log_erp_sd": np.log10(np.nanstd(erp["mean"], axis=0))})
    t = t.merge(probe_df[["channel_id", "shank", "y"]], on="channel_id")
    return flag_channels_local(t, col="log_erp_sd", radius_um=radius_um, thr_log=thr_log, only_flagged=False)

def erp_session(recs, trials, tmin=-1.0, tmax=1.0):
    """
    ERPs and ERP checks of one session.
    recs   : {"original": rec, "CMR": rec} (fn.load_session)
    trials : kept trials of the session (trials_filtered.csv, keep == True)
    Events : Press = press_neural_s (all); Press_TTL / Press_MA = press_ttl_np_s / press_np_s of the trials that have both;
             Tone = tone_neural_s of correct + late trials; Release = release_neural_s.
    Returns erps {(ref, event): erp dict}, sharp (one row per ref: TTL/MA median ratio, % channels TTL larger),
            outl (one row per channel x event, CMR: local_dev, flag_local).
    """
    both = trials["press_from_ttl"] & trials["press_np_s"].notna()
    ev = {"Press":     trials["press_neural_s"].values,
          "Press_TTL": trials.loc[both, "press_ttl_np_s"].values,
          "Press_MA":  trials.loc[both, "press_np_s"].values,
          "Tone":      trials.loc[trials["label"].isin(["correct", "late"]), "tone_neural_s"].values,
          "Release":   trials["release_neural_s"].values}
    erps = {(ref, name): event_erp(rec, ts, tmin=tmin, tmax=tmax) for ref, rec in recs.items() for name, ts in ev.items()}
    sharp = []
    for ref in recs:
        s = alignment_sharpness(erps[(ref, "Press_TTL")], erps[(ref, "Press_MA")])
        sharp.append({"ref": ref, "n_trials": erps[(ref, "Press_TTL")]["n"], "ttl_ma_ratio_median": s["ratio_a_b"].median(),
                      "pct_channels_ttl_larger": 100 * (s["ratio_a_b"] > 1).mean()})
    pdf = probe_table(recs["original"])
    outl = []
    for name in ["Press", "Tone", "Release"]:
        o = erp_outliers(erps[("CMR", name)], pdf)
        o.insert(0, "event", name)
        outl.append(o)
    return erps, pd.DataFrame(sharp), pd.concat(outl, ignore_index=True)

# ════════════════════════════════════════════════════════════════════════
# 5. BAND POWER (all channels, all bands, one file per session and reference)
#    own Morlet transform with scipy FFT; from lfp_main.process_session (2026-10-19/20), bursts removed,
#    new trial columns, all long_prob blocks
# ════════════════════════════════════════════════════════════════════════
import time as _time
import pickle as _pickle
import warnings as _warnings
from scipy import fft as _sfft

MAIN_BANDS = {                      # frequencies sampled inside each band; cycles as in the pilot
    "delta 1-3":    {"freqs": [1.0, 1.5, 2.0, 2.5, 3.0],          "n_cycles": 2},
    "theta 4-7":    {"freqs": [4.0, 5.0, 6.0, 7.0],               "n_cycles": 5},
    "alpha 8-12":   {"freqs": [8.0, 9.0, 10.0, 11.0, 12.0],       "n_cycles": 6},
    "beta 13-29":   {"freqs": [13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0], "n_cycles": 7},
    "lowG 30-60":   {"freqs": [30.0, 35.0, 40.0, 45.0, 55.0, 60.0], "n_cycles": 10},   # 50 Hz skipped (notch)
    "highG 61-100": {"freqs": [65.0, 72.0, 80.0, 88.0, 96.0],     "n_cycles": 10},
}
MAIN_EPOCHS  = {"Press": (-2.5, 3.5), "Release": (-3.0, 2.5)}        # long enough for 1 Hz wavelets
MAIN_CURVE_T = {"Press": (-0.7, 2.0), "Release": (-1.5, 0.6)}       # part kept for the time courses
MAIN_WINDOWS = {"pre":         ("Press",   (-0.5, -0.2)),
                "hold_early":  ("Press",   (0.10, 0.30)),
                "hold":        ("Press",   (0.20, 0.85)),
                "pre_release": ("Release", (-0.30, 0.00)),
                "post":        ("Release", (0.20, 0.50))}
TONE_WINDOWS = {"pre_tone": (-0.3, 0.0), "post_tone": (0.0, 0.3)}   # relative to the tone (Press epochs)
MAIN_PRE     = (-0.5, -0.2)                                         # own pre-press baseline of the time courses
SPEC_WINDOWS = {"pre": (-0.8, -0.2), "hold": (0.25, 0.85), "pre_tone": (-0.6, 0.0)}   # pre_tone relative to the tone


def session_trials(kept, session):
    """Kept trials of one session in the form band_power_session needs (neural clock, seconds)."""
    t = kept[kept["session"] == session]
    return pd.DataFrame({"trial_index": t["trial_index"].values, "label": t["label"].values, "type": t["type"].values,
                         "long_prob": t["long_prob"].values, "press_s": t["press_neural_s"].values,
                         "tone_s": t["tone_neural_s"].values, "release_s": t["release_neural_s"].values})


def main_conditions(tr):
    """Boolean masks of the conditions for the time courses and spectra: label x type, all long_prob blocks,
    plus 'early long (hold>0.3)' (early long trials held > 0.3 s, as in the old files) so both early-long versions exist."""
    conds = {f"{o} {t}": ((tr["label"] == o) & (tr["type"] == t)) for o in ("correct", "early", "late") for t in ("long", "short")}
    conds["early long (hold>0.3)"] = (tr["label"] == "early") & (tr["type"] == "long") & (tr["hold_s"] > 0.3)
    return conds


def morlet(f, n_cycles, fs):
    """Complex Morlet wavelet (sigma_t = n_cycles / (2 pi f), +-5 sigma), unit energy."""
    sigma = n_cycles / (2 * np.pi * f)
    t = np.arange(-5 * sigma, 5 * sigma + 1 / fs, 1 / fs)
    w = np.exp(2j * np.pi * f * t) * np.exp(-t ** 2 / (2 * sigma ** 2))
    return (w / np.linalg.norm(w)).astype(np.complex64)


def read_epochs(rec, event_s, tmin, tmax):
    """All channels around every event -> (n_events_ok, n_channels, n_samples) float32 + mask of events kept."""
    fs = rec.get_sampling_frequency()
    n_pre, n_post = int(round(-tmin * fs)), int(round(tmax * fs))
    n_tot, out, ok = rec.get_num_samples(), [], []
    for t in event_s:
        if not np.isfinite(t):
            ok.append(False); continue
        i = int(round(t * fs))
        if i - n_pre < 0 or i + n_post > n_tot:
            ok.append(False); continue
        out.append(rec.get_traces(start_frame=i - n_pre, end_frame=i + n_post, return_in_uV=True).T.astype(np.float32))
        ok.append(True)
    return np.stack(out), np.array(ok)


def _power(Xf, nfft, n, f, nc, fs, decim):
    """Power of one frequency for FFT'd data Xf (..., nfft) -> (..., n // decim) float32."""
    w = morlet(f, nc, fs)
    L = len(w)
    c = _sfft.ifft(Xf * _sfft.fft(w, nfft)[None, None, :], axis=-1, workers=-1)[..., L // 2: L // 2 + n]
    return (np.abs(c[..., ::decim]) ** 2).astype(np.float32)


def _session_spectra(X, fs, tr, conds):
    """Hann periodogram (1 Hz bins, 1-100 Hz) per window, mean over the trials of each condition.
    hold: only trials still holding at the window end; pre_tone: only trials with a tone heard while holding."""
    t0 = MAIN_EPOCHS["Press"][0]
    hold, td = tr["hold_s"].values, tr["tone_delay_s"].values
    out, cnt = {}, {}
    for w, (a, z) in SPEC_WINDOWS.items():
        nwin = int(round((z - a) * fs))
        han = np.hanning(nwin).astype(np.float32)
        S = np.full((X.shape[0], X.shape[1], 100), np.nan, np.float32)
        valid = np.zeros(X.shape[0], bool)
        for i in range(X.shape[0]):
            if w == "pre_tone":
                if not np.isfinite(td[i]) or td[i] > hold[i] + 1e-9:
                    continue
                a_i = td[i] + a
            else:
                a_i = a
                if w == "hold" and z > hold[i] + 1e-9:
                    continue
            i0 = int(round((a_i - t0) * fs))
            seg = X[i, :, i0:i0 + nwin]
            if seg.shape[-1] != nwin:
                continue
            F = np.fft.rfft((seg - seg.mean(axis=-1, keepdims=True)) * han, n=int(fs), axis=-1)
            S[i] = (np.abs(F[:, 1:101]) ** 2).astype(np.float32)
            valid[i] = True
        out[w], cnt[w] = {}, {}
        for c, mk in conds.items():
            sel = valid & mk.values
            cnt[w][c] = int(sel.sum())
            out[w][c] = np.nanmean(S[sel], axis=0) if sel.any() else np.full((X.shape[1], 100), np.nan, np.float32)
    return out, cnt


def band_power_session(rec, trials, out_file, decim=10, mem_bytes=2e9, verbose=True, meta_extra=None):
    """
    Band power of one session for one referenced recording (all good channels), saved to out_file (pickle).
    rec    : referenced recording (fn.load_session: CMR or LOCAL)
    trials : fn.session_trials (kept trials; press_s, tone_s, release_s on the neural clock)
    Saves:
      meta     channel_ids, x, y, fs, bands, windows, n_trials, decim, seconds (+ meta_extra)
      trials   trial table actually used (rows = rows of the arrays), incl. hold_s, tone_delay_s
      logpow   {band: {window: (trials, channels)}} log10 of the mean power in the window (band freqs x time);
               windows pre, hold_early, hold, pre_release, post, pre_tone, post_tone (NaN for trials without a tone)
      curves   {align: {band: {condition: (channels, times)}}} mean over trials of log10(power(t) / own pre-press power),
               each trial only while the paw is on the lever; counts {align: {condition: (times,)}}; times {align: (times,)}
      spectra  {window: {condition: (channels, 100)}} Hann periodogram 1-100 Hz, mean over trials; spec_n = trial counts
    Conditions: correct / early / late x long / short, all long_prob blocks (no hold filter).
    """
    t0 = _time.time()
    fs = rec.get_sampling_frequency()
    tr = trials.copy().reset_index(drop=True)
    tr["hold_s"] = tr["release_s"] - tr["press_s"]
    tr["tone_delay_s"] = tr["tone_s"] - tr["press_s"]                   # NaN for trials without a tone
    ep, ok = {}, {}
    for al, col in (("Press", "press_s"), ("Release", "release_s")):
        ep[al], ok[al] = read_epochs(rec, tr[col].values, *MAIN_EPOCHS[al])
    keep = ok["Press"] & ok["Release"]                                  # same trials in both alignments
    ep = {al: ep[al][keep[ok[al]]] for al in ep}                        # rows of ep[al] are the ok[al] trials
    tr = tr[keep].reset_index(drop=True)
    nT, nC = len(tr), ep["Press"].shape[1]
    conds = main_conditions(tr)
    hold, td = tr["hold_s"].values, tr["tone_delay_s"].values.astype(float)

    times = {al: (np.arange(ep[al].shape[-1]) / fs + MAIN_EPOCHS[al][0])[::decim] for al in ep}
    curve_m = {al: (times[al] >= MAIN_CURVE_T[al][0] - 1e-9) & (times[al] <= MAIN_CURVE_T[al][1] + 1e-9) for al in ep}
    on = {"Press": times["Press"][curve_m["Press"]][None, :] <= hold[:, None],
          "Release": times["Release"][curve_m["Release"]][None, :] >= -hold[:, None]}
    wins = list(MAIN_WINDOWS) + list(TONE_WINDOWS)
    logpow = {b: {w: np.full((nT, nC), np.nan, np.float32) for w in wins} for b in MAIN_BANDS}
    curves = {al: {b: {c: np.full((nC, curve_m[al].sum()), np.nan, np.float32) for c in conds} for b in MAIN_BANDS}
              for al in ep}
    tdu = np.unique(np.round(td[np.isfinite(td)], 2))                   # tone delays present (0.35, 1.05)
    spectra, spec_n = _session_spectra(ep["Press"], fs, tr, conds)

    n = {al: ep[al].shape[-1] for al in ep}
    Lmax = len(morlet(min(f for b in MAIN_BANDS.values() for f in b["freqs"]), 2, fs))
    nfft = {al: _sfft.next_fast_len(n[al] + Lmax) for al in ep}
    chunk = max(1, int(mem_bytes / (nT * max(nfft.values()) * 8 * 3)))
    tP = times["Press"]
    mpre = (tP >= MAIN_PRE[0] - 1e-9) & (tP <= MAIN_PRE[1] + 1e-9)
    for c0 in range(0, nC, chunk):
        cs = slice(c0, min(nC, c0 + chunk))
        Xf = {al: _sfft.fft(ep[al][:, cs, :], nfft[al], axis=-1, workers=-1) for al in ep}
        for b, spec in MAIN_BANDS.items():
            P = {al: np.stack([_power(Xf[al], nfft[al], n[al], f, spec["n_cycles"], fs, decim) for f in spec["freqs"]],
                              axis=2) for al in ep}                     # (trials, ch, freqs, times)
            for w, (al, (a, z)) in MAIN_WINDOWS.items():                # fixed windows
                m = (times[al] >= a - 1e-9) & (times[al] <= z + 1e-9)
                logpow[b][w][:, cs] = np.log10(P[al][..., m].mean(axis=(2, 3)))
            for v in tdu:                                               # tone windows, per tone delay
                rows = np.isclose(np.round(td, 2), v)
                for w, (a, z) in TONE_WINDOWS.items():
                    m = (tP >= v + a - 1e-9) & (tP <= v + z + 1e-9)
                    logpow[b][w][np.ix_(rows, np.arange(cs.start, cs.stop))] = np.log10(P["Press"][rows][..., m].mean(axis=(2, 3)))
            base = np.log10(P["Press"][..., mpre]).mean(axis=-1, keepdims=True)    # own pre-press, (trials, ch, freqs, 1)
            for al in ep:
                lr = (np.log10(P[al][..., curve_m[al]]) - base).mean(axis=2)        # (trials, ch, times)
                lr = np.where(on[al][:, None, :], lr, np.nan)                       # only while on the lever
                for c, mk in conds.items():
                    if mk.sum():
                        with np.errstate(invalid="ignore"), _warnings.catch_warnings():
                            _warnings.simplefilter("ignore", RuntimeWarning)
                            curves[al][b][c][cs] = np.nanmean(lr[mk.values], axis=0)
            del P
        del Xf
        if verbose:
            print(f"    channels {cs.start}-{cs.stop - 1} of {nC} done ({_time.time() - t0:.0f} s)", flush=True)

    counts = {al: {c: on[al][mk.values].sum(axis=0) for c, mk in conds.items()} for al in ep}
    loc = rec.get_channel_locations()
    res = {"meta": {"channel_ids": [str(c) for c in rec.get_channel_ids()], "x": loc[:, 0], "y": loc[:, 1], "fs": fs,
                    "bands": MAIN_BANDS, "windows": MAIN_WINDOWS, "tone_windows": TONE_WINDOWS,
                    "spectra_windows": SPEC_WINDOWS, "n_trials": nT, "decim": decim,
                    "seconds": round(_time.time() - t0, 1), **(meta_extra or {})},
           "trials": tr, "logpow": logpow, "curves": curves, "counts": counts,
           "times": {al: times[al][curve_m[al]] for al in ep}, "spectra": spectra, "spec_n": spec_n}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file + ".tmp", "wb") as f:
        _pickle.dump(res, f)
    os.replace(out_file + ".tmp", out_file)                             # file appears only when complete
    return res

# ════════════════════════════════════════════════════════════════════════
# 6. BURSTS (Karvat, Alyahyay & Diester 2021 PNAS; public code github.com/Optophys/Spontaneous)
#    from analysis_main cell 2.6 (2026-10-21): same rule; beta exactly as before, gamma added (10 cycles);
#    theta and NeuroDSP dropped; one representative channel per region (PL, IL, Cg1)
# ════════════════════════════════════════════════════════════════════════
BURST_BANDS = {"beta 15-30":   {"range": (15, 30),  "n_cycles": 7,  "skip": None},       # exactly as before
               "gamma 30-100": {"range": (30, 100), "n_cycles": 10, "skip": (48, 52)}}   # 48-52 Hz: notch, never a peak
BURST_PCT, BURST_ART_UV, BURST_ART_PAD, BURST_BIN = 90, 1000.0, 500, 0.05
BURST_ALIGN = {"press": ("press_s", -1.5, 2.0), "tone": ("tone_s", -1.0, 0.5), "release": ("release_s", -1.5, 0.5)}
REGION_FULL = {"PL": "Prelimbic area", "IL": "Infralimbic area", "Cg1": "Cingulate area 1"}


def read_full(rec, ids, chunk_s=60):
    """Whole session for a few channels, read in chunks -> (samples, channels) float64."""
    fs, n = rec.get_sampling_frequency(), rec.get_num_samples()
    step = int(chunk_s * fs)
    return np.concatenate([rec.get_traces(start_frame=a, end_frame=min(n, a + step), channel_ids=ids, return_in_uV=True)
                           for a in range(0, n, step)], axis=0).astype(np.float64)


def _ndi_dilate(mask, pad):
    """Mark pad samples on each side of every True sample (artifact padding)."""
    import scipy.ndimage as ndi
    return ndi.binary_dilation(mask, structure=np.ones(2 * pad + 1, bool))


def karvat_bursts(x, fs, band, n_cycles, art, skip=None, pct=BURST_PCT):
    """
    Burst-on (bool per sample) and burst peak events (bool per sample), Karvat et al. 2021 steps:
    1 Morlet power (n_cycles), 1-Hz steps, band ±1 Hz, whole session   2 artifact samples -> NaN
    3 / pct-th percentile per frequency; < 1 -> 0, NaN -> 0             4 2-D regional maxima (8-connectivity); edge rows dropped
    5 extent = samples around the peak where power at the peak frequency stays above the percentile
    skip = (lo, hi) Hz rows set to 0 (no peak there; e.g. 50 Hz notch).
    """
    import scipy.signal as ss
    from skimage.morphology import local_maxima
    foi = np.arange(max(1, band[0] - 1), band[1] + 2)                  # band ±1 Hz, 1-Hz steps
    P = np.empty((len(foi), len(x)), np.float32)
    for i, f in enumerate(foi):
        P[i] = np.abs(ss.fftconvolve(x, morlet(f, n_cycles, fs), mode="same")) ** 2
    P[:, art] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        M = P / np.nanpercentile(P, pct, axis=1, keepdims=True)
    del P
    M[~(M >= 1)] = 0                                                    # < percentile and NaN -> 0
    if skip is not None:
        M[(foi >= skip[0]) & (foi <= skip[1])] = 0
    pk = local_maxima(M, connectivity=2, allow_borders=True) & (M > 0)  # = MATLAB imregionalmax (8-connectivity)
    pk[[0, -1], :] = False                                              # ±1 Hz rows only used for peak detection
    on, ev = np.zeros(len(x), bool), np.zeros(len(x), bool)
    for i in range(1, len(foi) - 1):
        js = np.flatnonzero(pk[i])
        if not len(js):
            continue
        row = M[i] > 0
        run = np.cumsum(np.diff(np.concatenate([[0], row.astype(np.int8)])) == 1)
        on |= np.isin(run, np.unique(run[js])) & row                    # extent at the peak frequency
        ev[js] = True
    return on, ev


def burst_binned(on, ev, fs, times, a, z, bin_s=BURST_BIN):
    """(trials, bins): occupancy (fraction of the bin in a burst) and number of burst peaks, around event times (s)."""
    nb, w = int(round((z - a) / bin_s)), int(round(bin_s * fs))
    occ = np.full((len(times), nb), np.nan, np.float32); cnt = occ.copy()
    cs, es = np.concatenate([[0], np.cumsum(on)]), np.concatenate([[0], np.cumsum(ev)])
    for k, t0 in enumerate(times):
        if not np.isfinite(t0):
            continue
        st = int(round((t0 + a) * fs)) + np.arange(nb) * w
        if st[0] < 0 or st[-1] + w > len(on):
            continue
        occ[k], cnt[k] = (cs[st + w] - cs[st]) / w, es[st + w] - es[st]
    return occ, cnt


def pick_region_channels(rec, session, rep, chreg, rat):
    """One channel per region: the representative position of the rat (rep: rat, region, x, y; the 1.13b rule);
    if that position is not recorded in this session, the nearest channel of the same region (distance returned)."""
    ids = np.array([str(c) for c in rec.get_channel_ids()])
    loc = rec.get_channel_locations()
    reg = chreg[chreg["session"] == session].set_index("channel_id")["region"].reindex(ids).values
    pick, dist_um = {}, {}
    for _, rr in rep[rep["rat"].astype(str) == str(rat)].iterrows():
        d = np.hypot(loc[:, 0] - rr["x"], loc[:, 1] - rr["y"])
        d[reg != REGION_FULL.get(rr["region"], rr["region"])] = np.inf     # only channels of the same region
        if np.isfinite(d.min()):
            k = int(np.argmin(d)); pick[rr["region"]] = ids[k]; dist_um[rr["region"]] = float(d[k])
    return pick, dist_um


def bursts_session(recs, trials, rep, chreg, session, rat, out_file, verbose=True):
    """
    Karvat 2021 bursts (BURST_BANDS) for one session, every reference in recs ({"CMR": rec, "LOCAL": rec}),
    one representative channel per region. Saved to out_file (.npz) + trial table (<out_file>_trials.csv):
      <ref>|<region>|channel, |distance_um (0 = representative position), |artifact_frac
      <ref>|<band>|<region>|session_occ                 fraction of the whole session (no artifacts) in a burst
      <ref>|<band>|<region>|<align>|occ / |cnt          (trials, 50-ms bins) around press, tone, release (BURST_ALIGN)
    """
    t0 = _time.time()
    res = {}
    for ref, rec in recs.items():
        fs = rec.get_sampling_frequency()
        pick, dist_um = pick_region_channels(rec, session, rep, chreg, rat)
        if not pick:
            continue
        X = read_full(rec, list(pick.values()))
        for j, (reg, cid) in enumerate(pick.items()):
            x = X[:, j]
            art = _ndi_dilate(np.abs(x) > BURST_ART_UV, BURST_ART_PAD)
            res[f"{ref}|{reg}|channel"] = np.array([cid])
            res[f"{ref}|{reg}|distance_um"] = np.array([dist_um[reg]])
            res[f"{ref}|{reg}|artifact_frac"] = np.array([art.mean()])
            for band, cfg in BURST_BANDS.items():
                on, ev = karvat_bursts(x, fs, cfg["range"], cfg["n_cycles"], art, skip=cfg["skip"])
                key = f"{ref}|{band}|{reg}"
                res[key + "|session_occ"] = np.array([on[~art].mean()])
                for al, (col, a, z) in BURST_ALIGN.items():
                    res[f"{key}|{al}|occ"], res[f"{key}|{al}|cnt"] = burst_binned(on, ev, fs, trials[col].values.astype(float), a, z)
                if verbose:
                    print(f"    {ref} {reg} {band}: occupancy {on[~art].mean():.3f} ({_time.time() - t0:.0f} s)", flush=True)
        del X
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    trials.to_csv(out_file.replace(".npz", "_trials.csv"), index=False)
    np.savez_compressed(out_file.replace(".npz", ".tmp.npz"), **res)
    os.replace(out_file.replace(".npz", ".tmp.npz"), out_file)          # file appears only when complete
    return res

# ════════════════════════════════════════════════════════════════════════
# 7. STATISTICS
# ════════════════════════════════════════════════════════════════════════
def sign_flip_p(values, n_perm=10000, seed=0):
    """Two-sided sign-flip test of mean(values) == 0 across units (sessions). Returns (mean, p, n).
    (from lfp_main, unchanged)"""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(np.mean(v)) if len(v) else np.nan), np.nan, len(v)
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, len(v))) * v).mean(axis=1)
    return float(v.mean()), float((1 + np.sum(np.abs(null) >= abs(v.mean()))) / (n_perm + 1)), len(v)

def m1_session(r, bands, min_trials=10):
    """M1 for one band-power file: correct long trials, per channel mean over trials of log10(hold / pre) and
    log10(hold / post), then median over channels. Returns {band: {"hold-pre": v, "hold-post": v}} or None if too few trials."""
    T, lp = r["trials"], r["logpow"]
    cl = ((T["label"] == "correct") & (T["type"] == "long")).values
    if cl.sum() < min_trials:
        return None, int(cl.sum())
    out = {}
    for b in bands:
        L = lp[b]
        out[b] = {"hold-pre":  float(np.median((L["hold"][cl] - L["pre"][cl]).mean(axis=0))),
                  "hold-post": float(np.median((L["hold"][cl] - L["post"][cl]).mean(axis=0)))}
    return out, int(cl.sum())

def m2_session(r, bands, min_cl=10, min_el=5, early_min_hold=0.3, windows=("pre_release", "hold_early")):
    """M2 for one band-power file: correct long minus early long (hold > early_min_hold), per window relative to each trial's
    own pre-press: per channel mean over trials of log10(window / pre) in each group, difference, median over channels.
    Returns ({band: {window: value}}, n_correct_long, n_early_long) or (None, n, n) if too few trials."""
    T, lp = r["trials"], r["logpow"]
    cl = ((T["label"] == "correct") & (T["type"] == "long")).values
    el = ((T["label"] == "early") & (T["type"] == "long") & (T["hold_s"] > early_min_hold)).values
    if cl.sum() < min_cl or el.sum() < min_el:
        return None, int(cl.sum()), int(el.sum())
    out = {}
    for b in bands:
        L = lp[b]
        out[b] = {w: float(np.median((L[w][cl] - L["pre"][cl]).mean(0) - (L[w][el] - L["pre"][el]).mean(0))) for w in windows}
    return out, int(cl.sum()), int(el.sum())

def session_curve(r, align, band, cond, min_trials=5, min_on=3):
    """Time course of one condition in one band-power file: median over channels of the saved condition curve
    (log10 power / own pre-press, trials only while on the lever). None if the condition has < min_trials trials;
    time points with < min_on trials still holding -> NaN."""
    mk = main_conditions(r["trials"])[cond]
    if mk.sum() < min_trials:
        return None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        cur = np.nanmedian(r["curves"][align][band][cond], axis=0)
    return np.where(r["counts"][align][cond] >= min_on, cur, np.nan)

REGIONS = {"PL": "Prelimbic area", "IL": "Infralimbic area", "Cg1": "Cingulate area 1"}


def region_masks(r, chreg, session, regions=REGIONS):
    """{region: bool mask over the channels of a band-power file}, from channel_regions_all.csv (01_sessions_setup, P1).
    Channels not in the table -> no region. Bad channels are already missing from the band-power files."""
    c = chreg[chreg["session"] == session].set_index("channel_id")["region"]
    reg = c.reindex([str(x) for x in r["meta"]["channel_ids"]]).values
    return {k: (reg == v) for k, v in regions.items()}, reg


def region_session_values(r, masks, bands, min_cl=10, min_el=5, min_tr=5, min_ch=10):
    """All per-region session values of the old cell 1.12 for one band-power file.
    Returns a list of dicts (measure, region, band, value_log10). Value = mean over trials, median over the region's channels."""
    T, C = r["trials"], main_conditions(r["trials"])
    cl, el = C["correct long"].values, C["early long (hold>0.3)"].values
    tP, tR = r["times"]["Press"], r["times"]["Release"]
    pre_tone = {"short": (0.05, 0.35), "long": (0.75, 1.05)}
    out = []
    def add(meas, reg, b, v):
        out.append({"measure": meas, "region": reg, "band": b, "value_log10": float(v)})
    def med(x, m):
        return np.median(np.nanmean(x[:, m], axis=0))
    with _warnings.catch_warnings(), np.errstate(invalid="ignore"):
        _warnings.simplefilter("ignore", RuntimeWarning)
        for reg, m in masks.items():
            if m.sum() < min_ch:
                continue
            for b in bands:
                L = r["logpow"][b]
                if cl.sum() >= min_cl:                                              # M1, M1b
                    add("M1 hold − pre (CL)", reg, b, med(L["hold"][cl] - L["pre"][cl], m))
                    add("M1 hold − post (CL)", reg, b, med(L["hold"][cl] - L["post"][cl], m))
                    add("M1b hold − hold_early (CL)", reg, b, med(L["hold"][cl] - L["hold_early"][cl], m))
                for cname, mk in (("CL", cl), ("EL", el)):                         # Q1
                    if mk.sum() >= min_tr:
                        add(f"Q1 pre_release − pre ({cname})", reg, b, med(L["pre_release"][mk] - L["pre"][mk], m))
                if cl.sum() >= min_cl and el.sum() >= min_el:                       # M2
                    for w in ("pre_release", "hold_early"):
                        dC = np.nanmean(L[w][cl] - L["pre"][cl], axis=0)
                        dE = np.nanmean(L[w][el] - L["pre"][el], axis=0)
                        add(f"M2 {w} correct − early", reg, b, np.median((dC - dE)[m]))
                if cl.sum() >= 10 and el.sum() >= min_tr:                           # Q4a, time-matched
                    cP = np.nanmedian(r["curves"]["Press"][b]["correct long"][m], axis=0)
                    cP = np.where(r["counts"]["Press"]["correct long"] >= 3, cP, np.nan)
                    eR = np.nanmedian(r["curves"]["Release"][b]["early long (hold>0.3)"][m], axis=0)
                    for w in [(-0.3, 0.0), (-0.45, -0.15)]:
                        e_val = np.nanmean(eR[(tR >= w[0]) & (tR <= w[1])])
                        c_val = np.nanmean([np.nanmean(cP[(tP >= h + w[0]) & (tP <= h + w[1])])
                                            for h in T.loc[el, "hold_s"].values])
                        add(f"Q4a early − correct, matched {w}", reg, b, e_val - c_val)
                for typ, (a, z) in pre_tone.items():                                # step A
                    cC, cLt = f"correct {typ}", f"late {typ}"
                    if C[cC].sum() < 5 or C[cLt].sum() < 5:
                        continue
                    mt = (tP >= a - 1e-9) & (tP <= z + 1e-9)
                    if not ((r["counts"]["Press"][cC][mt] >= 3).all() and (r["counts"]["Press"][cLt][mt] >= 3).all()):
                        continue
                    Cc = np.nanmedian(r["curves"]["Press"][b][cC][m], axis=0)
                    Lc = np.nanmedian(r["curves"]["Press"][b][cLt][m], axis=0)
                    add(f"stepA late − correct before tone ({typ})", reg, b, np.nanmean(Lc[mt]) - np.nanmean(Cc[mt]))
    return out


def sign_flip_table(vals, bands, rat_of_session):
    """vals: DataFrame with session, measure, region, band, value_log10 -> sign-flip per measure x region x band x rat (+ all)."""
    d = vals.copy(); d["rat"] = d["session"].map(rat_of_session)
    out = []
    for (meas, reg), g in d.groupby(["measure", "region"], sort=False):
        for b in bands:
            for rat in ["all"] + sorted(g["rat"].unique()):
                v = g[(g["band"] == b) & ((g["rat"] == rat) if rat != "all" else True)]["value_log10"].dropna()
                m, p, n = sign_flip_p(v.values)
                out.append({"measure": meas, "region": reg, "band": b, "rat": rat, "n_sessions": n,
                            "sessions_gt0": int((v > 0).sum()), "mean_%": round(100 * (10 ** m - 1), 1) if n else np.nan,
                            "p": round(p, 4) if n >= 2 else np.nan})
    return pd.DataFrame(out)

def ref_compare_values(r, chreg, session, ref, bands, min_cl=10, min_ch=10, min_span_mm=0.2,
                       measures=(("hold − pre", "hold", "pre"), ("hold − post", "hold", "post"))):
    """Old cell 1.19 for one band-power file (one reference): correct long trials; per channel mean over trials of
    log10(window1 / window0); per region (>= min_ch channels): median over channels, % channels > 0, linear slope of the
    channel % change vs DV (per mm, if the region spans >= min_span_mm); plus IL − PL. Returns a list of dicts."""
    cl = main_conditions(r["trials"])["correct long"].values
    if cl.sum() < min_cl:
        return []
    ids = [str(x) for x in r["meta"]["channel_ids"]]
    c = chreg[chreg["session"] == session].set_index("channel_id")
    R, D = c["region"].reindex(ids).values, c["dv_mm"].reindex(ids).values.astype(float)
    pct = lambda v: 100 * (10 ** np.asarray(v, float) - 1)
    rows = []
    for b in bands:
        L = r["logpow"][b]
        for meas, w1, w0 in measures:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                ch = np.nanmean(L[w1][cl] - L[w0][cl], axis=0)
            vals = {}
            for reg, name in REGIONS.items():
                m = (R == name) & np.isfinite(ch)
                if m.sum() < min_ch:
                    continue
                ok = m & np.isfinite(D)
                slope = (np.polyfit(D[ok], pct(ch[ok]), 1)[0]
                         if ok.sum() >= min_ch and np.ptp(D[ok]) >= min_span_mm else np.nan)
                vals[reg] = float(np.median(ch[m]))
                rows.append(dict(ref=ref, session=session, band=b, measure=meas, region=reg, n_ch=int(m.sum()),
                                 med_log=vals[reg], pct_ch_up=100 * float((ch[m] > 0).mean()), slope_pct_mm=slope))
            if "IL" in vals and "PL" in vals:
                rows.append(dict(ref=ref, session=session, band=b, measure=meas, region="IL − PL", n_ch=np.nan,
                                 med_log=vals["IL"] - vals["PL"], pct_ch_up=np.nan, slope_pct_mm=np.nan))
    return rows

# ════════════════════════════════════════════════════════════════════════
# 8. BURST RESULTS (from analysis_main cells 3.0 / 3.1, 2026-10-22), for the files of bursts_session
# ════════════════════════════════════════════════════════════════════════
BURST_A0 = {al: a for al, (_, a, _) in BURST_ALIGN.items()}          # start of each saved window (s)


def burst_window(O, align, a, z, bin_s=BURST_BIN):
    """Mean occupancy per trial in [a, z] s around the event, from the saved (trials, 50-ms bins) array."""
    i0, i1 = int(round((a - BURST_A0[align]) / bin_s)), int(round((z - BURST_A0[align]) / bin_s))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(O[:, max(i0, 0):i1], axis=1)


def burst_trial_tables(z, T, band, refs=("CMR", "LOCAL"), regions=("PL", "IL", "Cg1"),
                       pre_tone=None, w_rel=(-0.75, -0.15), min_rt=None):
    """
    Trial tables for the burst tests of one session (old cell 3.0).
    z : loaded .npz of bursts_session; T : its _trials.csv
    late table : correct and late trials, occupancy in the window before the TONE (long −0.6..0, short −0.3..0 s)
    early table: early long trials (hold >= 0.75 s, so the window lies inside the hold) in the window −0.75..−0.15 s before
                 their release, and correct long trials (a) time-matched (same time after the press as each early trial's
                 window, averaged over the early holds) and (b) before their own release.
    Groups: each region and 'mPFC pooled' (mean over the regions of the trial). min_rt: optional minimum RT for correct trials.
    """
    pre_tone = pre_tone or {"long": (-0.6, 0.0), "short": (-0.3, 0.0)}
    T = T.copy()
    T["hold_s"], T["rt_s"] = T["release_s"] - T["press_s"], T["release_s"] - T["tone_s"]
    C = {k: v.values for k, v in main_conditions(T).items()}
    rt, hold = T["rt_s"].values, T["hold_s"].values
    rt_ok = np.ones(len(T), bool) if min_rt is None else (rt >= min_rt)
    cl = np.where(C["correct long"] & rt_ok)[0]
    el = np.where(C["early long"] & (hold >= -w_rel[0]))[0]
    late, early = [], []
    for ref in refs:
        key = {reg: f"{ref}|{band}|{reg}" for reg in regions if f"{ref}|{band}|{reg}|tone|occ" in z.files}
        if not key:
            continue
        def pooled(d):
            d = dict(d); d["mPFC pooled"] = np.nanmean(np.stack(list(d.values())), axis=0); return d
        for typ, (a, b) in pre_tone.items():
            idx = np.where((C[f"correct {typ}"] & rt_ok) | C[f"late {typ}"])[0]
            if not len(idx):
                continue
            o = pooled({reg: burst_window(z[k + "|tone|occ"], "tone", a, b)[idx] for reg, k in key.items()})
            for g, v in o.items():
                late.append(pd.DataFrame(dict(ref=ref, group=g, typ=typ, trial=idx, cond=T["label"].values[idx],
                                              rt=rt[idx], occ=v)))
        if len(el) and len(cl):
            e_ = pooled({reg: burst_window(z[k + "|release|occ"], "release", *w_rel)[el] for reg, k in key.items()})
            own = pooled({reg: burst_window(z[k + "|release|occ"], "release", *w_rel)[cl] for reg, k in key.items()})
            tmx = pooled({reg: np.nanmean(np.stack([burst_window(z[k + "|press|occ"], "press", h + w_rel[0], h + w_rel[1])[cl]
                                                    for h in hold[el]]), axis=0) for reg, k in key.items()})
            for g in e_:
                for cond, tri, v in (("early long", el, e_[g]), ("correct long, time-matched", cl, tmx[g]),
                                     ("correct long, own release", cl, own[g])):
                    early.append(pd.DataFrame(dict(ref=ref, group=g, typ="long", trial=tri, cond=cond, hold=hold[tri], occ=v)))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    return cat(late), cat(early)


def burst_session_values(df, ref, group, typ, a, b, min_tr=1):
    """One value per session: 100 x (mean occupancy in condition a - in condition b), sessions with >= min_tr trials in each."""
    d = df[(df["ref"] == ref) & (df["group"] == group) & (df["typ"] == typ) & df["cond"].isin([a, b])]
    if d.empty:
        return pd.DataFrame(columns=["session", "rat", "value", "n_a", "n_b"])
    g = d.groupby(["session", "rat", "cond"])["occ"].agg(["mean", "count"]).unstack("cond")
    if ("mean", a) not in g or ("mean", b) not in g:
        return pd.DataFrame(columns=["session", "rat", "value", "n_a", "n_b"])
    ok = (g[("count", a)] >= min_tr) & (g[("count", b)] >= min_tr)
    g = g[ok]
    return pd.DataFrame({"session": g.index.get_level_values(0), "rat": g.index.get_level_values(1),
                         "value": 100 * (g[("mean", a)] - g[("mean", b)]).values,
                         "n_a": g[("count", a)].values, "n_b": g[("count", b)].values})