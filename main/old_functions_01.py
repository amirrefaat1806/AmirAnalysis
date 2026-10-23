"""
lfp_main.py  -  new functions for the main analysis (pilot + analysis_main).
The pilot files tfr_functions_pilot.py / tfr_stats_pilot.py stay frozen; everything new goes here.

Sections
  1. Signal QC   (PSD per channel, outlier channels, 50 Hz check, depth x frequency maps, raw traces)
  2. Event check (event-locked averages / ERPs per channel, depth maps, TTL vs MA alignment)
  3. Main analysis (all sessions): band power per trial x channel x window + time courses per condition,
     one file per session (process_session); sign-flip test across sessions (sign_flip_p)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


# ======================================================================
# 1. SIGNAL QC
def compute_psd(rec, channel_ids=None, n_segments=30, seg_s=10.0, fmax=100.0, nperseg_s=2.0, seed=0):
    """
    Welch PSD per channel from `n_segments` pieces of `seg_s` seconds spread evenly over the whole
    recording (reading all 40 min x 377 channels at once would need several GB of RAM).
    Returns (freqs, psd, channel_ids): psd = (n_channels, n_freqs) in uV^2/Hz, mean over segments.
    """
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
    return f[keep], psd[:, keep], channel_ids


def psd_table(freqs, psd, channel_ids, probe_df, bands=None, line_hz=50.0):
    """
    One row per channel: shank, x, y, log10 band power, broadband (1-100 Hz) log power,
    deviation from the shank median in MAD units, and the 50 Hz line ratio
    (power at 49-51 Hz / power at 45-48 + 52-55 Hz; ~1 = no line left after the notch).
    """
    bands = bands or {"delta": (1, 3), "theta": (4, 7), "alpha": (8, 12), "beta": (13, 29),
                      "low_gamma": (30, 60), "high_gamma": (61, 100)}
    df = pd.DataFrame({"channel_id": channel_ids})
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


def flag_channels(table, mad_thr=5.0, line_thr=2.0):
    """Channels whose broadband power is > mad_thr MAD from their shank median, or with a 50 Hz line.
    NOTE: after CMR whole depth blocks can pass this threshold (real depth structure) -> use
    flag_channels_local for single bad channels."""
    t = table.copy()
    t["flag_power"] = t["broadband_dev_mad"].abs() > mad_thr
    t["flag_line"] = t["line_ratio"] > line_thr
    return t[t["flag_power"] | t["flag_line"]].sort_values(["shank", "y"])


def flag_channels_local(table, col="log_broadband", radius_um=75.0, thr_log=0.3):
    """
    Single-channel outliers: each channel vs the median of its neighbours on the same shank
    (|y difference| <= radius_um, the channel itself excluded). thr_log = 0.3 log10 units = factor 2.
    Depth blocks (several neighbouring channels together) are NOT flagged, only lone odd channels.
    Returns the table with 'local_dev' and 'flag_local' columns, flagged rows only.
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
    return t[t["flag_local"]].sort_values(["shank", "y"])


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


def plot_psd_depth(freqs, psd, channel_ids, probe_df, title, save_path=None, show=True):
    """
    Per shank, two panels: (top) log10 PSD, depth x frequency; (bottom) same minus the shank's
    median spectrum, so single odd channels stand out. y axis true to scale (gaps in the map are
    left empty/white); deepest (small y = near the tip) at the bottom.
    """
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


def _save_show(fig, save_path, show):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ======================================================================
# 2. EVENT CHECK (event-locked averages)
def event_erp(rec, event_times_s, channel_ids=None, tmin=-1.0, tmax=1.0):
    """
    Event-locked average (ERP) for every channel, without keeping all trials in memory.
    event_times_s: times in s on the recording clock (NaN and windows off the recording are skipped).
    Returns dict: mean (n_times, n_ch), sem (n_times, n_ch), times, n (trials used), channel_ids.
    """
    fs = rec.get_sampling_frequency()
    channel_ids = list(rec.get_channel_ids() if channel_ids is None else channel_ids)
    n_pre, n_post = int(round(-tmin * fs)), int(round(tmax * fs))
    n_tot, n_samp = n_pre + n_post, rec.get_num_samples()
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
        x -= x[:n_pre].mean(axis=0, keepdims=True)          # remove the pre-event mean (per trial)
        s1 = x if s1 is None else s1 + x
        s2 = x ** 2 if s2 is None else s2 + x ** 2
        n += 1
    mean = s1 / n
    sem = np.sqrt(np.maximum(s2 / n - mean ** 2, 0) / max(n - 1, 1))
    return {"mean": mean, "sem": sem, "times": (np.arange(n_tot) - n_pre) / fs, "n": n,
            "channel_ids": channel_ids}


def plot_erp_depth(erp, probe_df, title, vmax=None, marks=None, save_path=None, show=True):
    """
    Per shank: channels (true-scale depth, deepest at the bottom) x time, colour = ERP (µV).
    Top row: median over all channels (the part shared by the whole probe).
    marks = {label: time_s} extra dashed lines (e.g. tone delay).
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


# ======================================================================
# 3. MAIN ANALYSIS: band power per trial x channel x window, and time courses per condition
#    (all channels, all bands, one file per session; own Morlet transform with numpy/scipy FFT)
import time as _time
import pickle as _pickle
from scipy import fft as _sfft

MAIN_BANDS = {                      # frequencies sampled inside each band; cycles as in the pilot CONFIG
    "delta 1-3":    {"freqs": [1.0, 1.5, 2.0, 2.5, 3.0],          "n_cycles": 2},
    "theta 4-7":    {"freqs": [4.0, 5.0, 6.0, 7.0],               "n_cycles": 5},
    "alpha 8-12":   {"freqs": [8.0, 9.0, 10.0, 11.0, 12.0],       "n_cycles": 6},
    "beta 13-29":   {"freqs": [13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0], "n_cycles": 7},
    "lowG 30-60":   {"freqs": [30.0, 35.0, 40.0, 45.0, 55.0, 60.0], "n_cycles": 10},   # 50 Hz skipped (notch)
    "highG 61-100": {"freqs": [65.0, 72.0, 80.0, 88.0, 96.0],     "n_cycles": 10},
}
MAIN_EPOCHS = {"Press": (-2.5, 3.5), "Release": (-3.0, 2.5)}          # long enough for 1 Hz wavelets (+-1.6 s)
MAIN_CURVE_T = {"Press": (-0.7, 2.0), "Release": (-1.5, 0.6)}         # part kept for the time courses
MAIN_WINDOWS = {"pre":         ("Press",   (-0.5, -0.2)),
                "hold_early":  ("Press",   (0.10, 0.30)),
                "hold":        ("Press",   (0.20, 0.85)),
                "pre_release": ("Release", (-0.30, 0.00)),
                "post":        ("Release", (0.20, 0.50))}
MAIN_PRE = (-0.5, -0.2)


def main_conditions(tr):
    """Boolean masks of the curve conditions (long_prob 0.5 blocks only)."""
    b = tr["long_prob"] == 0.5
    o, t, h = tr["outcome"], tr["type"], tr["hold_s"]
    return {"correct long": b & (o == "correct") & (t == "long"),
            "correct short": b & (o == "correct") & (t == "short"),
            "late long": b & (o == "late") & (t == "long"),
            "late short": b & (o == "late") & (t == "short"),
            "early long (hold>0.3)": b & (o == "early") & (t == "long") & (h > 0.3),
            "early short": b & (o == "early") & (t == "short")}


def morlet(f, n_cycles, fs):
    """Complex Morlet wavelet (sigma_t = n_cycles / (2 pi f), +-5 sigma), unit energy."""
    sigma = n_cycles / (2 * np.pi * f)
    t = np.arange(-5 * sigma, 5 * sigma + 1 / fs, 1 / fs)
    w = np.exp(2j * np.pi * f * t) * np.exp(-t ** 2 / (2 * sigma ** 2))
    return (w / np.linalg.norm(w)).astype(np.complex64)


def read_epochs(rec, event_s, tmin, tmax):
    """All channels around every event -> (n_events, n_channels, n_samples) float32 + mask of events kept."""
    fs = rec.get_sampling_frequency()
    n_pre, n_post = int(round(-tmin * fs)), int(round(tmax * fs))
    n_tot, out, ok = rec.get_num_samples(), [], []
    for t in event_s:
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


V2_BURST_BANDS = ("theta 4-7", "beta 13-29")
V2_BURST_RANGE = (-1.5, 3.0)            # s from press used for thresholds and detection
V2_BURST_PCT, V2_BURST_MIN_S = 98.0, 0.070
V2_SPEC_WINDOWS = {"pre": (-0.8, -0.2), "hold": (0.25, 0.85), "pre_tone": (-0.6, 0.0)}   # pre_tone rel. to tone


def _detect_bursts(P, t, freqs, fs_p, ch0):
    """P (trials, ch, freqs, times) power at fs_p Hz, t times (s from press). Returns DataFrame of bursts."""
    m = (t >= V2_BURST_RANGE[0] - 1e-9) & (t <= V2_BURST_RANGE[1] + 1e-9)
    P, t = P[..., m], t[m]
    thr = np.percentile(P, V2_BURST_PCT, axis=(0, 3), keepdims=True)          # (1, ch, freqs, 1)
    R = P / thr
    lo = np.concatenate([np.full_like(P[:, :, :1], -np.inf), P[:, :, :-1]], axis=2)
    hi = np.concatenate([P[:, :, 1:], np.full_like(P[:, :, :1], -np.inf)], axis=2)
    ok = (R > 1) & (P >= lo) & (P >= hi)                                       # above threshold + local max in freq
    Rm = np.where(ok, R, 0).max(axis=2)                                        # (trials, ch, times)
    fi = np.where(ok, R, 0).argmax(axis=2)
    mask = Rm > 0
    pad = np.zeros(mask.shape[:2] + (1,), bool)
    d = np.diff(np.concatenate([pad, mask, pad], axis=2).astype(np.int8), axis=2)
    st, en = np.argwhere(d == 1), np.argwhere(d == -1)                         # same lexicographic order
    if not len(st):
        return pd.DataFrame(columns=["trial", "ch", "t_start", "t_end", "peak_ratio", "f_center"])
    k0, k1 = st[:, 2], en[:, 2]                                                # end index is exclusive
    keep = (k1 - k0) / fs_p >= V2_BURST_MIN_S - 1e-9
    st, k0, k1 = st[keep], k0[keep], k1[keep]
    nT_, nC_, nt = Rm.shape
    flat = Rm.reshape(-1)
    base = (st[:, 0] * nC_ + st[:, 1]) * nt
    idx = np.empty(2 * len(k0), np.int64); idx[0::2] = base + k0; idx[1::2] = base + k1 - 1
    peak = np.maximum.reduceat(flat, idx)[0::2] if len(k0) else np.array([])
    peak = np.maximum(peak, flat[base + k1 - 1])                               # include last sample of the run
    kc = (k0 + k1 - 1) // 2
    return pd.DataFrame({"trial": st[:, 0].astype(np.int32), "ch": (st[:, 1] + ch0).astype(np.int32),
                         "t_start": t[k0].astype(np.float32), "t_end": (t[k1 - 1] + 1 / fs_p).astype(np.float32),
                         "peak_ratio": peak.astype(np.float32),
                         "f_center": freqs[fi[st[:, 0], st[:, 1], kc]].astype(np.float32)})


def _session_spectra(X, fs, tr, conds, td):
    """Hann periodogram (1 Hz bins, 1-100 Hz) per window, mean over the trials of each condition."""
    t0 = MAIN_EPOCHS["Press"][0]
    hold = tr["hold_s"].values
    out, cnt = {}, {}
    for w, (a, z) in V2_SPEC_WINDOWS.items():
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


def process_session(rec, trials, out_file, decim=10, mem_bytes=6e8, verbose=True, v2=False):
    """
    v2=True (band_power_v2) additionally saves, all from the Press-aligned epochs:
      logpow[band]["pre_tone"/"post_tone"]  tone −0.3..0 / 0..+0.3 (needs column tone_s; NaN for trials without tone)
      spectra {window: {condition: (channels, 100)}}  Hann periodogram 1–100 Hz (1 Hz bins), mean over trials,
              windows pre −0.8..−0.2, hold 0.25..0.85, pre_tone tone−0.6..tone (only trials still holding); spec_n counts
      bursts  {band: DataFrame}  theta 4-7 and beta 13-29 bursts (Karvat et al. 2020 offline rule): power at a
              frequency > its 98th percentile (per channel x frequency, over all trials, press −1.5..+3.0 s) and a
              local maximum across frequency; ≥ 70 ms. Columns trial (row in trials), ch (index), t_start, t_end
              (s from press), peak_ratio (max power/threshold), f_center (Hz at the burst centre)
    rec    : referenced recording (e.g. CMR), all good channels
    trials : kept trials of the session; columns trial, outcome, type, long_prob, press_s, release_s
    Saves to out_file (pickle):
      meta      channel_ids, x, y, fs, bands, windows, n_trials
      trials    trial table (trials used, in the order of the arrays) incl. hold_s
      logpow    {band: {window: (trials, channels)}}  log10 of the mean power in the window (band freqs x time)
      curves    {align: {band: {condition: (channels, times)}}}  mean over trials of log10(power(t)/own pre-press),
                each trial only while on the lever; counts {align: {condition: (times,)}}; times {align: (times,)}
    """
    t0 = _time.time()
    fs = rec.get_sampling_frequency()
    tr = trials.copy().reset_index(drop=True)
    tr["hold_s"] = tr["release_s"] - tr["press_s"]
    if v2:
        tr["tone_delay_s"] = (tr["tone_s"] - tr["press_s"]) if "tone_s" in tr else np.nan
    ep, ok = {}, {}
    for al, col in (("Press", "press_s"), ("Release", "release_s")):
        ep[al], ok[al] = read_epochs(rec, tr[col].values, *MAIN_EPOCHS[al])
    keep = ok["Press"] & ok["Release"]                                    # same trials in both alignments
    ep = {al: ep[al][keep[ok[al]]] for al in ep}                          # rows of ep[al] are the ok[al] trials
    tr = tr[keep].reset_index(drop=True)
    nT, nC = len(tr), ep["Press"].shape[1]
    conds = main_conditions(tr)
    hold = tr["hold_s"].values

    times = {al: (np.arange(ep[al].shape[-1]) / fs + MAIN_EPOCHS[al][0])[::decim] for al in ep}
    curve_m = {al: (times[al] >= MAIN_CURVE_T[al][0] - 1e-9) & (times[al] <= MAIN_CURVE_T[al][1] + 1e-9) for al in ep}
    on = {"Press": times["Press"][curve_m["Press"]][None, :] <= hold[:, None],
          "Release": times["Release"][curve_m["Release"]][None, :] >= -hold[:, None]}
    logpow = {b: {w: np.full((nT, nC), np.nan, np.float32) for w in MAIN_WINDOWS} for b in MAIN_BANDS}
    curves = {al: {b: {c: np.full((nC, curve_m[al].sum()), np.nan, np.float32) for c in conds} for b in MAIN_BANDS}
              for al in ep}

    if v2:
        td = tr["tone_delay_s"].values.astype(float)
        for b in MAIN_BANDS:
            logpow[b]["pre_tone"] = np.full((nT, nC), np.nan, np.float32)
            logpow[b]["post_tone"] = np.full((nT, nC), np.nan, np.float32)
        tdu = [v for v in np.unique(np.round(td[np.isfinite(td)], 2))]
        bursts = {b: [] for b in V2_BURST_BANDS}
        spectra, spec_n = _session_spectra(ep["Press"], fs, tr, conds, td)

    n = {al: ep[al].shape[-1] for al in ep}
    Lmax = len(morlet(min(f for b in MAIN_BANDS.values() for f in b["freqs"]), 2, fs))
    nfft = {al: _sfft.next_fast_len(n[al] + Lmax) for al in ep}
    chunk = max(1, int(mem_bytes / (nT * max(nfft.values()) * 8 * 3)))
    for c0 in range(0, nC, chunk):
        cs = slice(c0, min(nC, c0 + chunk))
        Xf = {al: _sfft.fft(ep[al][:, cs, :], nfft[al], axis=-1, workers=-1) for al in ep}
        for b, spec in MAIN_BANDS.items():
            pw = {al: [] for al in ep}                                      # per freq (trials, ch, times)
            for f in spec["freqs"]:
                for al in ep:
                    pw[al].append(_power(Xf[al], nfft[al], n[al], f, spec["n_cycles"], fs, decim))
            P = {al: np.stack(pw[al], axis=2) for al in ep}                 # (trials, ch, freqs, times)
            # windows: log10 of mean power (freqs x time)
            for w, (al, (a, z)) in MAIN_WINDOWS.items():
                m = (times[al] >= a - 1e-9) & (times[al] <= z + 1e-9)
                logpow[b][w][:, cs] = np.log10(P[al][..., m].mean(axis=(2, 3)))
            if v2:
                tP = times["Press"]
                for v in tdu:                                               # trials with this tone delay
                    rows = np.isclose(np.round(td, 2), v)
                    for w, (a, z) in (("pre_tone", (v - 0.3, v)), ("post_tone", (v, v + 0.3))):
                        m = (tP >= a - 1e-9) & (tP <= z + 1e-9)
                        logpow[b][w][np.ix_(rows, np.arange(cs.start, cs.stop))] = \
                            np.log10(P["Press"][rows][..., m].mean(axis=(2, 3)))
                if b in V2_BURST_BANDS:
                    bursts[b].append(_detect_bursts(P["Press"], tP, np.asarray(spec["freqs"]), fs / decim, cs.start))
            # curves: mean log power per freq relative to own pre-press (mean of log), band mean, on lever only
            mpre = (times["Press"] >= MAIN_PRE[0] - 1e-9) & (times["Press"] <= MAIN_PRE[1] + 1e-9)
            base = np.log10(P["Press"][..., mpre]).mean(axis=-1, keepdims=True)    # (trials, ch, freqs, 1)
            for al in ep:
                lr = (np.log10(P[al][..., curve_m[al]]) - base).mean(axis=2)        # (trials, ch, times)
                lr = np.where(on[al][:, None, :], lr, np.nan)
                for c, mk in conds.items():
                    if mk.sum():
                        with np.errstate(invalid="ignore"), warnings_off():
                            curves[al][b][c][cs] = np.nanmean(lr[mk.values], axis=0)
            del P, pw
        if verbose:
            print(f"    channels {cs.start}-{cs.stop - 1} of {nC} done ({_time.time() - t0:.0f} s)", flush=True)

    counts = {al: {c: on[al][mk.values].sum(axis=0) for c, mk in conds.items()} for al in ep}
    loc = rec.get_channel_locations()
    res = {"meta": {"channel_ids": list(rec.get_channel_ids()), "x": loc[:, 0], "y": loc[:, 1], "fs": fs,
                    "bands": MAIN_BANDS, "windows": MAIN_WINDOWS, "n_trials": nT, "decim": decim,
                    "seconds": round(_time.time() - t0, 1)},
           "trials": tr, "logpow": logpow, "curves": curves, "counts": counts,
           "times": {al: times[al][curve_m[al]] for al in ep}}
    if v2:
        res["bursts"] = {b: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for b, v in bursts.items()}
        res["spectra"], res["spec_n"] = spectra, spec_n
        res["meta"]["v2"] = {"burst_rule": "Karvat 2020 offline: >98th pct per channel x freq (press -1.5..3.0 s), "
                                           "local max across freq, >=70 ms", "burst_bands": list(V2_BURST_BANDS),
                             "spectra_windows": V2_SPEC_WINDOWS}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file + ".tmp", "wb") as f:
        _pickle.dump(res, f)
    os.replace(out_file + ".tmp", out_file)                               # file appears only when complete
    return res


class warnings_off:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings(); self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
    def __exit__(self, *a):
        self._w.__exit__(*a)


def sign_flip_p(values, n_perm=10000, seed=0):
    """Two-sided sign-flip test of mean(values) == 0 across units (sessions). Returns (mean, p, n)."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(np.mean(v)) if len(v) else np.nan), np.nan, len(v)
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, len(v))) * v).mean(axis=1)
    return float(v.mean()), float((1 + np.sum(np.abs(null) >= abs(v.mean()))) / (n_perm + 1)), len(v)