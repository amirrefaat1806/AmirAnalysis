"""
tfr_functions.py  -  loading, time-frequency extraction, baselines and across-channel plots.

Sections
  1. Loading                (recording, behaviour)
  2. Extraction             (Morlet TFR per band, trial-averaged)
  3. Baselines              (own / event / common pooled baseline)
  4. Probe + across-channel plots
  5. Titles and save paths
  6. Sign-flip result plots (probe heatmap, significance map)

The permutation tests live in tfr_stats.py (imported there as `tfs`).
"""

import os

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import spikeinterface.full as si
from matplotlib.patches import Polygon

DEFAULT_YTICKS = [1, 4, 8, 13, 20, 30, 40, 50, 60, 70, 80, 90, 99]


# ======================================================================
# 1. LOADING
def load_recording(path):
    """Load a (down-sampled) recording folder with spikeinterface."""
    return si.load(path)


def _read_behavior(session_dir, csv_rel_path):
    """trials.csv + tone_np_s (tone time on the recording clock; NaN when there was no tone)."""
    beh = pd.read_csv(str(session_dir) + csv_rel_path)
    beh["tone_np_s"] = np.where(beh["tone"] > 0,
                                (beh["tone"] - beh["press"]) + beh["press_np_s"], np.nan)
    if "release_np_s" not in beh.columns:
        beh["release_np_s"] = (beh["release"] - beh["press"]) + beh["press_np_s"]
    return beh


def load_behavior(session_dir, csv_rel_path, outcome, trial_type):
    """Returns (all trials, trials with beh[outcome]==1 and beh[trial_type]==1)."""
    beh = _read_behavior(session_dir, csv_rel_path)
    filtered = beh[(beh[outcome] == 1) & (beh[trial_type] == 1)]
    print(f"Total trials: {len(beh)} | {outcome}: {(beh[outcome] == 1).sum()} | "
          f"{trial_type}: {(beh[trial_type] == 1).sum()} | both: {len(filtered)}")
    return beh, filtered


def load_pooled_trials(session_dir, csv_rel_path, conditions=None):
    """All trials with a press (conditions=None) or the union of a list like [("correct","short"), ...]."""
    beh = _read_behavior(session_dir, csv_rel_path)
    if conditions is None:
        pooled = beh.dropna(subset=["press_np_s"])
    else:
        pooled = pd.concat([beh[(beh[o] == 1) & (beh[t] == 1)] for o, t in conditions])
        pooled = pooled[~pooled.index.duplicated()]
    print(f"Pooled trials: {len(pooled)}")
    return pooled


# ======================================================================
# 2. EXTRACTION + PER-BAND TFR (trial-averaged, raw power, no baseline yet)
def extract_raw_power(ds_data, trials_df, channel_id, events, band_settings, tmin, tmax, fs):
    """
    One channel. For every event in `events` ({name: behaviour column in s}) cut the window
    [tmin, tmax] around the event and compute the Morlet TFR separately for every band
    (each band has its own n_cycles). Returns {event: [AverageTFR per band]}.
    """
    n_pre, n_post = round(fs * abs(tmin)), round(fs * tmax)
    n_total = n_pre + n_post
    raw_power = {}
    for name, column in events.items():
        arr = []
        for t_sec in trials_df[column].dropna().values:
            idx = int(t_sec * fs)
            trace = ds_data.get_traces(start_frame=idx - n_pre, end_frame=idx + n_post,
                                       channel_ids=[channel_id], return_in_uV=True)
            if trace.shape[0] == n_total:               # drop windows that run off the recording
                arr.append(trace[:, 0])
        info = mne.create_info(ch_names=[channel_id], sfreq=fs, ch_types="seeg")
        epochs = mne.EpochsArray(np.array(arr)[:, np.newaxis, :], info, tmin=tmin, verbose=False)
        raw_power[name] = [
            epochs.compute_tfr(method="morlet", freqs=b["freqs"], n_cycles=b["n_cycles"],
                               use_fft=True, return_itc=False, average=True, decim=1,
                               n_jobs=1, verbose=False)
            for b in band_settings]
        print(f"{name}: {len(arr)} valid trials")
    return raw_power


# ======================================================================
# 3. BASELINES (log10 ratio, results stacked over bands -> (freqs, times))
def _stack(results, band_settings):
    freqs = np.concatenate([b["freqs"] for b in band_settings])
    return {k: np.concatenate(v, axis=0) for k, v in results.items()}, freqs


def compute_own_baseline(raw_power, band_settings, baseline_window, crop_window):
    """Every event is baselined against its OWN pre-event window. Returns (results, times, freqs)."""
    results = {n: [] for n in raw_power}
    times = None
    for b in range(len(band_settings)):
        for name in raw_power:
            p = raw_power[name][b].copy()
            p.apply_baseline(baseline=baseline_window, mode="logratio")
            p.crop(tmin=crop_window[0], tmax=crop_window[1])
            results[name].append(p.data[0])
            times = p.times
    results, freqs = _stack(results, band_settings)
    return results, times, freqs


def get_baseline_values(ref_power_list, baseline_window):
    """Mean power per frequency inside the baseline window; ref_power_list = raw_power[event]."""
    vals = []
    for p in ref_power_list:
        m = (p.times >= baseline_window[0]) & (p.times <= baseline_window[1])
        vals.append(p.data[:, :, m].mean(axis=-1, keepdims=True))
    return vals


def compute_common_baseline(raw_power, band_settings, baseline_vals, crop_window, mode="log10"):
    """Baseline with values computed elsewhere. mode='log10' or 'pct' (percentage change)."""
    results = {n: [] for n in raw_power}
    times = None
    for b in range(len(band_settings)):
        for name in raw_power:
            p = raw_power[name][b]
            m = (p.times >= crop_window[0]) & (p.times <= crop_window[1])
            ratio = p.data / baseline_vals[b]
            if mode == "pct":
                results[name].append((ratio - 1)[0][:, m] * 100)
            else:
                results[name].append(np.log10(ratio)[0][:, m])
            times = p.times[m]
    results, freqs = _stack(results, band_settings)
    return results, times, freqs

def compute_event_baseline(raw_power, band_settings, baseline_window, crop_window, baseline_event):
    """Every event baselined against ONE reference event's pre-window (e.g. Press)."""
    base = {b: get_baseline_values([raw_power[baseline_event][b]], baseline_window)[0]
            for b in range(len(band_settings))}
    results = {n: [] for n in raw_power}
    times = None
    for b in range(len(band_settings)):
        for name in raw_power:
            p = raw_power[name][b]
            m = (p.times >= crop_window[0]) & (p.times <= crop_window[1])
            results[name].append(np.log10(p.data / base[b])[0][:, m])
            times = p.times[m]
    results, freqs = _stack(results, band_settings)
    return results, times, freqs


# ======================================================================
# 4. PROBE LAYOUT + ACROSS-CHANNEL PLOT
def extract_and_baseline_multichannel(ds_data, trials_df, channel_ids, events, band_settings,
                                      tmin, tmax, fs, baseline_window, crop_window, baseline_event):
    """extract_raw_power + compute_event_baseline for every channel -> ({channel: results}, times, freqs)."""
    all_results, times, freqs = {}, None, None
    for ch in channel_ids:
        raw = extract_raw_power(ds_data, trials_df, ch, events, band_settings, tmin, tmax, fs)
        all_results[ch], times, freqs = compute_event_baseline(
            raw, band_settings, baseline_window, crop_window, baseline_event)
        print(f"Done: {ch}")
    return all_results, times, freqs


def get_probe_df(ds_data):
    """Probe geometry (x, y, ...) with the matching channel_id of every contact."""
    probe = ds_data.get_probe()
    df = probe.to_dataframe().copy()
    ids = ds_data.get_channel_ids()
    df["channel_id"] = [ids[i] for i in probe.device_channel_indices]
    return df


def select_every_nth_channel(probe_df, n):
    """Every n-th contact of the probe table (n=20 -> every 20th)."""
    return probe_df.iloc[::n].reset_index(drop=True)


def draw_probe(ax, probe_df, x, y, contour=None):
    """Mini probe diagram: grey contacts, red dot = this channel."""
    if contour is not None:
        ax.add_patch(Polygon(contour, closed=True, facecolor="#f2f2f2", edgecolor="#888888",
                             linewidth=1.0, zorder=0))
    ax.scatter(probe_df["x"], probe_df["y"], s=4, marker="s", color="#b0b0b0",
               edgecolors="none", zorder=2)
    ax.scatter(x, y, s=20, color="red", edgecolors="black", linewidths=1, zorder=10)
    ax.set_xlim(probe_df["x"].min() - 100, probe_df["x"].max() + 700)
    ax.set_ylim(probe_df["y"].min() - 150, probe_df["y"].max() + 150)
    ax.tick_params(labelsize=6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def percent_ticks(cbar, vmax, pct=(-50, -25, 0, 50, 100, 200, 300)):
    """Label a log10-ratio colour bar in percent change."""
    pct = np.array(pct)
    pos = np.log10(1 + pct / 100)
    keep = (pos >= -vmax) & (pos <= vmax)
    cbar.set_ticks(pos[keep])
    cbar.set_ticklabels([f"{p:+d}%" if p != 0 else "0%" for p in pct[keep]])


def plot_across_channels(all_results, times, freqs, event_names, selected_df, probe_df,
                         baseline_event_name, title, freq_range=(1, 99), yticks=None,
                         band_boundaries=None, save_path=None, vmax=None, contour=None,
                         vlines=None, mode="log10",show=True):
    """
    One row per channel of selected_df, one column per entry of event_names
    (events or conditions), probe location in the last column.
    all_results[channel_id][name] = (freqs, times) array. vlines = {name: seconds} red dashed.
    """
    yticks = yticks or DEFAULT_YTICKS
    n_ch = len(selected_df)
    if vmax is None:
        vals = np.concatenate([all_results[c][n].ravel() for c in all_results for n in event_names])
        vmax = np.nanmax(np.abs(vals[np.isfinite(vals)]))
    n_cols = len(event_names) + 1
    fig, axes = plt.subplots(n_ch, n_cols, figsize=(5.5 * n_cols, 3.2 * n_ch), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * len(event_names) + [0.8],
                                          "wspace": 0.18, "hspace": 0.45})
    im = None
    for row, (_, ch_row) in enumerate(selected_df.iterrows()):
        ch = ch_row["channel_id"]
        for col, name in enumerate(event_names):
            ax = axes[row, col]
            im = ax.imshow(all_results[ch][name], aspect="auto", origin="lower",
                           extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap="viridis",
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
            ax.axvline(0, color="white", linestyle="--", linewidth=1.2)
            if vlines and vlines.get(name) is not None:
                ax.axvline(vlines[name], color="red", linestyle="--", linewidth=1.5)
            for f in (band_boundaries or []):
                ax.axhline(f, color="white", linestyle="--", linewidth=0.6)
            ax.set_ylim(freq_range)
            ax.set_yticks(yticks)
            ax.set_xlim(times[0], times[-1])
            ax.grid(False)
            ax.tick_params(axis="both", which="both", length=3, labelsize=7)
            ax.set_xlabel("Time (s)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Frequency (Hz)", fontsize=8)
                ax.text(-0.22, 0.5, ch, transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=14, fontweight="bold")
            else:
                ax.tick_params(axis="y", labelleft=False)
        draw_probe(axes[row, -1], probe_df, ch_row["x"], ch_row["y"], contour)
    for col, name in enumerate(event_names):
        axes[0, col].set_title(name, fontsize=15, fontweight="bold", pad=10)
    axes[0, -1].set_title("Probe Location", fontsize=13, fontweight="bold", pad=10)
    fig_h = 3.2 * n_ch
    fig.suptitle(title, fontsize=20, fontweight="bold", y=1 - 0.1 / fig_h, va="top")
    plt.subplots_adjust(left=0.08, right=0.93, top=1 - 1.0 / fig_h, bottom=0.025,
                        wspace=0.15, hspace=0.45)
    cbar = fig.colorbar(im, ax=axes[:, :len(event_names)], location="right", pad=0.01,
                        fraction=0.03, aspect=40, extend="both")
    if mode == "pct":
        ticks = np.linspace(-vmax, vmax, 9)   # ends always labelled: -100 ... +100
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:+.0f}%" if t != 0 else "0%" for t in ticks])
        cbar.set_label("% change from baseline", fontsize=10)
    else:
        percent_ticks(cbar, vmax)
        cbar.set_label("Power change vs. baseline (%)", fontsize=10)

    cbar.ax.tick_params(labelsize=8)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ======================================================================
# 5. TITLES AND SAVE PATHS
def make_title(config, extra=""):
    n = config.get("n_trials")
    n_txt = f", n={n}" if n is not None else ""
    return (f"TFR - {config['outcome']} / {config['trial_type']} trials "
            f"({config.get('analysis_label', '')}{n_txt}) {extra}").strip()


def make_save_path(config, suffix):
    """<save_dir>/<outcome>_<trial_type>_<suffix>.png, or None when saving is off."""
    if not config.get("save_dir") or not config.get("save", False):
        return None
    return os.path.join(str(config["save_dir"]),
                        f"{config['outcome']}_{config['trial_type']}_{suffix}.png")


def make_save_path_named(config, name, folder=None):
    """<save_dir>[/<folder>]/<name>.png, or None when saving is off."""
    if not config.get("save_dir") or not config.get("save", False):
        return None
    parts = [str(config["save_dir"])] + ([folder] if folder else []) + [name + ".png"]
    return os.path.join(*parts)


def theta_depression_per_condition(ds_data, config, fs, conditions, baseline_window=(-0.5, -0.2),
                                   theta_range=(4, 8), time_window=(-0.05, 0.5)):
    """
    For each condition, compute theta power % change from a shared baseline
    (pre-press, all trials), averaged across all channels.

    Parameters
    ----------
    ds_data : recording object
    config : dict with keys 'session_dir', 'behavior_csv', 'events', 'band_settings',
             'tmin', 'tmax', 'crop_window', 'baseline_window'
    fs : sampling rate
    conditions : list of (outcome, trial_type) tuples, e.g. [("correct", "short"), ...]
    baseline_window : tuple (start, end) in seconds for pre-press baseline
    theta_range : tuple (fmin, fmax) in Hz
    time_window : tuple (start, end) in seconds for plotting

    Returns
    -------
    result : dict {label: 1D array of mean theta % change over time}
    plot_times : 1D array of time points
    n_trials : dict {label: number of trials}
    """
    press_only = {"Press": config["events"]["Press"]}
    probe_df = get_probe_df(ds_data)
    all_channels = probe_df["channel_id"].tolist()

    # Load all trials for shared baseline
    all_trials = load_pooled_trials(config["session_dir"], config["behavior_csv"], None)

    # Load trials per condition
    cond_trials = {}
    for o, t in conditions:
        label = f"{o} {t}"
        _, trials = load_behavior(config["session_dir"], config["behavior_csv"], o, t)
        if len(trials) > 0:
            cond_trials[label] = trials
        else:
            print(f"  {label}: 0 trials — skipping")

    # Loop through all channels
    all_power = {label: [] for label in cond_trials}
    times = None
    freqs = None

    for i, ch in enumerate(all_channels):
        ref = extract_raw_power(ds_data, all_trials, ch, press_only,
                                config["band_settings"], config["tmin"], config["tmax"], fs)
        base = get_baseline_values(ref["Press"], config["baseline_window"])

        for label, trials in cond_trials.items():
            raw = extract_raw_power(ds_data, trials, ch, press_only,
                                    config["band_settings"], config["tmin"], config["tmax"], fs)
            res, t, f = compute_common_baseline(raw, config["band_settings"], base,
                                                config["crop_window"])
            all_power[label].append(res["Press"])

            if times is None:
                times = np.array(t)
                freqs = np.array(f)

        if (i + 1) % 50 == 0:
            print(f"  done {i + 1}/{len(all_channels)} channels")

    print(f"  done all {len(all_channels)} channels")

    theta_mask = (freqs >= theta_range[0]) & (freqs <= theta_range[1])
    time_mask = (times >= time_window[0]) & (times <= time_window[1])
    plot_times = times[time_mask]

    result = {}
    n_trials_out = {}
    for label, ch_list in all_power.items():
        stacked = np.stack(ch_list)
        result[label] = stacked[:, theta_mask, :][:, :, time_mask].mean(axis=(0, 1))
        n_trials_out[label] = len(cond_trials[label])

    return result, plot_times, n_trials_out


# ======================================================================
# 6. SIGN-FLIP RESULT PLOTS (probe heatmap, significance map)

def plot_probe_layout(probe_df, selected_df=None, figsize=(4, 12)):
    """Show probe with channel positions. Selected channels are highlighted and labeled."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(probe_df["x"], probe_df["y"], s=6, c="#b0b0b0", edgecolors="none")
    if selected_df is not None:
        for _, row in selected_df.iterrows():
            ax.scatter(row["x"], row["y"], s=20, c="red", edgecolors="black", linewidths=0.5, zorder=5)
            ax.text(row["x"] + 40, row["y"], row["channel_id"], fontsize=5, va="center")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    n_sel = len(selected_df) if selected_df is not None else 0
    ax.set_title(f"Probe layout ({len(probe_df)} channels, {n_sel} selected)",
                 fontsize=11, fontweight="bold")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_sign_flip_heatmap(results_df, probe_df, title, band_order=None, save_path=None):
    """
    Probe heatmap from sign-flip results: one subplot per band showing the probe shape
    with each channel colored by mean % change. Bands side by side.
    """
    band_order = band_order or ["delta", "theta", "alpha", "beta", "gamma"]
    bands = [b for b in band_order if b in results_df["band"].values]
    n = len(bands)

    merged = results_df.merge(probe_df[["channel_id", "x", "y"]],
                               left_on="channel", right_on="channel_id")
    vmax = np.nanpercentile(np.abs(merged["mean_pct"].values), 95)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 1, 10), squeeze=False)
    axes = axes[0]
    sc = None
    for ax, band in zip(axes, bands):
        bd = merged[merged["band"] == band]
        n_sig = bd["significant"].sum()
        sc = ax.scatter(bd["x"], bd["y"], c=bd["mean_pct"], cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, s=8, edgecolors="none")
        ax.set_title(f"{band}\n{n_sig}/{len(bd)} sig", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(labelsize=6)
        if ax != axes[0]:
            ax.set_yticklabels([])
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.04, label="Mean % change")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_sign_flip_significance(results_df, probe_df, title, band_order=None, save_path=None):
    """
    Significance map: one subplot per band. Channels colored red (UP), blue (DOWN), grey (NS).
    Subtitle shows per-band UP/DOWN counts and means.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    band_order = band_order or ["delta", "theta", "alpha", "beta", "gamma"]
    bands = [b for b in band_order if b in results_df["band"].values]
    n = len(bands)
    n_total = results_df["channel"].nunique()

    merged = results_df.merge(probe_df[["channel_id", "x", "y"]],
                               left_on="channel", right_on="channel_id")
    cmap = ListedColormap(["#4477AA", "#CCCCCC", "#CC3311"])

    subtitle_parts = []
    for band in bands:
        bd = results_df[results_df["band"] == band]
        sig = bd[bd["significant"]]
        n_up = len(sig[sig["direction"] == "UP"])
        n_down = len(sig[sig["direction"] == "DOWN"])
        mean_up = sig.loc[sig["direction"] == "UP", "mean_pct"].mean() if n_up > 0 else 0
        mean_down = sig.loc[sig["direction"] == "DOWN", "mean_pct"].mean() if n_down > 0 else 0
        parts = []
        if n_up > 0:
            parts.append(f"UP:{n_up} ({mean_up:+.1f}%)")
        if n_down > 0:
            parts.append(f"DOWN:{n_down} ({mean_down:+.1f}%)")
        n_sig = n_up + n_down
        subtitle_parts.append(f"{band}: {n_sig}/{n_total} | {' | '.join(parts) if parts else 'none'}")

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 1, 10), squeeze=False)
    axes = axes[0]
    for ax, band in zip(axes, bands):
        bd = merged[merged["band"] == band]
        colors = []
        for _, row in bd.iterrows():
            if row["significant"] and row["direction"] == "UP":
                colors.append(2)
            elif row["significant"] and row["direction"] == "DOWN":
                colors.append(0)
            else:
                colors.append(1)
        ax.scatter(bd["x"], bd["y"], c=colors, cmap=cmap, vmin=-0.5, vmax=2.5,
                   s=8, edgecolors="none")
        n_sig = bd["significant"].sum()
        ax.set_title(f"{band}\n{n_sig}/{len(bd)} sig", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(labelsize=6)
        if ax != axes[0]:
            ax.set_yticklabels([])

    full_title = title + "\n" + " | ".join(subtitle_parts[:3])
    if len(subtitle_parts) > 3:
        full_title += "\n" + " | ".join(subtitle_parts[3:])
    fig.suptitle(full_title, fontsize=9, fontweight="bold", y=1.04)

    legend_elements = [Patch(facecolor="#CC3311", label="Significant UP"),
                       Patch(facecolor="#4477AA", label="Significant DOWN"),
                       Patch(facecolor="#CCCCCC", label="Not significant")]
    axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()