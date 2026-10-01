'''
Permutation-based statistics (Mik X Cohen book chapter 15) to compare TWO conditions
inside each alignment (Press / Tone / Release).
It works together with tfr_functions.py (imported as tfx) and does NOT change it.

Workflow:
  1. prepare_trial_data()      -> single-trial log-ratio power per condition and event
  2. sanity checks             -> check_overlap / check_shuffle_logic / false_positive_check
  3. run_permutation_stats()   -> real difference, null distribution, z-map, thresholds
  4. plots                     -> plot_permutation_maps / plot_permutation_bands
'''

import numpy as np                      # arrays and maths
import matplotlib.pyplot as plt         # plots
import mne                              # time-frequency (Morlet)
from scipy.stats import norm            # z <-> p conversion (like normcdf / norminv in MATLAB)

import tfr_functions as tfx             # your existing functions (load_behavior, load_pooled_trials, ...)


# ======================================================================
# 1. SINGLE-TRIAL POWER (no averaging over trials, otherwise we cannot shuffle trials)
def extract_single_trial_power(ds_data, trials_df, channel_id, event_column,
                               band_settings, tmin, tmax, fs,
                               keep_window=None, decim=1):
    """
    Like tfx.extract_raw_power, but with average=False, so we keep ONE power map
    per trial. Returns:
      power     : array (n_trials, n_freqs_total, n_times)  raw power (no baseline yet)
      times     : time axis (s) relative to the event
      freqs     : all band frequencies stitched together
      kept_ids  : index of the trials_df rows that were really used (valid trials)
    keep_window : (t0, t1) -> keep only this time range (saves a lot of memory)
    decim       : keep every decim-th time sample (e.g. 10) -> faster and lighter
    """
    n_pre = round(fs * abs(tmin))                    # samples before the event
    n_post = round(fs * tmax)                        # samples after the event
    n_total = n_pre + n_post                         # expected length of every snippet

    event_times = trials_df[event_column].dropna()   # event times (s); trials without this event are dropped
    arr_list = []                                    # collects the raw snippet of every valid trial
    kept_ids = []                                    # collects the trial index of every valid trial

    for trial_id, t_sec in event_times.items():      # loop over (trial index, event time in s)
        idx = int(t_sec * fs)                        # seconds -> sample number
        start, end = idx - n_pre, idx + n_post       # window around the event
        trace = ds_data.get_traces(start_frame=start, end_frame=end,
                                   channel_ids=[channel_id], return_in_uV=True)
        if trace.shape[0] == n_total:                # only keep snippets with the exact expected length
            arr_list.append(trace[:, 0])             # 1D signal of this trial
            kept_ids.append(trial_id)                # remember WHICH trial it was

    if len(arr_list) == 0:                           # nothing valid -> stop with a clear message
        raise ValueError(f"No valid trials for event column '{event_column}'")

    arr_3 = np.array(arr_list)[:, np.newaxis, :]     # (trials, 1 channel, time) as MNE expects
    info = mne.create_info(ch_names=[channel_id], sfreq=fs, ch_types='seeg')   # channel metadata
    epochs = mne.EpochsArray(arr_3, info, tmin=tmin, verbose=False)            # MNE epochs container

    power_per_band = []                              # one array per band
    times = None
    for band in band_settings:                       # delta, theta, alpha, beta, gamma
        tfr = epochs.compute_tfr(method='morlet', freqs=band['freqs'],
                                 n_cycles=band['n_cycles'], use_fft=True,
                                 return_itc=False, average=False,   # average=False -> keep single trials
                                 decim=decim, n_jobs=1, verbose=False)
        data = tfr.data[:, 0, :, :]                  # (trials, freqs, times) of the single channel
        times = tfr.times                            # time axis of this TFR
        if keep_window is not None:                  # optional cropping right away
            m = (times >= keep_window[0]) & (times <= keep_window[1])
            data = data[:, :, m]
            times = times[m]
        power_per_band.append(data.astype(np.float32))   # float32 halves the memory

    power = np.concatenate(power_per_band, axis=1)   # stitch bands along the frequency axis
    freqs = np.concatenate([b['freqs'] for b in band_settings])   # matching frequency axis
    return power, times, freqs, kept_ids


# ======================================================================
# 2. BASELINE FOR SINGLE TRIALS
def compute_pooled_baseline(power_pooled, times, baseline_window):
    """
    Mean raw power per frequency inside the baseline window, averaged over ALL pooled
    trials and over time -> array (n_freqs, 1). Same idea as tfx.get_baseline_values.
    """
    mask = (times >= baseline_window[0]) & (times <= baseline_window[1])   # time points of the baseline
    base = power_pooled[:, :, mask].mean(axis=(0, 2), dtype=np.float64)    # mean over trials and time
    return base[:, np.newaxis]                                             # (freqs, 1)


def apply_log_baseline(power, baseline_vals):
    """log10(power / baseline) for EVERY trial (single-trial log ratio)."""
    ratio = power / baseline_vals[np.newaxis, :, :].astype(np.float32)     # broadcast over trials and time
    return np.log10(ratio).astype(np.float32)                              # log ratio, (trials, freqs, times)


# ======================================================================
# 3. PREPARE EVERYTHING (the slow part: TFR). Run once, then reuse for all checks/tests.
def prepare_trial_data(ds_data, session_dir, csv_rel_path, conditions, channel_id,
                       events, band_settings, tmin, tmax, fs,
                       baseline_window, crop_window, baseline_event, decim=10):
    """
    conditions = [("late","short"), ("correct","long")]  (exactly two)
    1) pooled baseline from the trials of BOTH conditions (same as your common baseline)
    2) single-trial log-ratio power for each condition and each event
    Returns a dict 'prep' with: data {event: (data_A, data_B)}, labels, n_valid, times, freqs, ...
    """
    if len(conditions) != 2:                         # the test compares exactly two groups
        raise ValueError("conditions must contain exactly 2 entries")

    # --- pooled baseline (one baseline for both conditions) ---
    pooled_df = tfx.load_pooled_trials(session_dir, csv_rel_path, conditions)
    pooled_power, p_times, freqs, pooled_ids = extract_single_trial_power(
        ds_data, pooled_df, channel_id, events[baseline_event],
        band_settings, tmin, tmax, fs, keep_window=baseline_window, decim=decim)
    baseline_vals = compute_pooled_baseline(pooled_power, p_times, baseline_window)
    del pooled_power                                 # free memory, only the baseline values are needed

    # --- trials of each condition ---
    trial_dfs, labels = [], []
    for outcome, trial_type in conditions:
        _, trials = tfx.load_behavior(session_dir, csv_rel_path, outcome, trial_type)
        trial_dfs.append(trials)                     # table of the trials of this condition
        labels.append(f"{outcome} / {trial_type}")   # label for titles/legends
    check_overlap(trial_dfs[0], trial_dfs[1], labels)   # a trial must not be in both conditions

    # --- single-trial log-ratio power per event and condition ---
    data, n_valid = {}, {}
    times = None
    for ev_name, ev_col in events.items():           # Press, Tone, Release
        per_cond = []
        for trials in trial_dfs:                     # condition A, then condition B
            power, times, freqs, kept = extract_single_trial_power(
                ds_data, trials, channel_id, ev_col, band_settings,
                tmin, tmax, fs, keep_window=crop_window, decim=decim)
            per_cond.append(apply_log_baseline(power, baseline_vals))   # (trials, freqs, times)
        data[ev_name] = (per_cond[0], per_cond[1])
        n_valid[ev_name] = (per_cond[0].shape[0], per_cond[1].shape[0])  # REAL n after dropping invalid trials
        print(f"{ev_name}: {labels[0]} n={n_valid[ev_name][0]} | "
              f"{labels[1]} n={n_valid[ev_name][1]} | shape A={per_cond[0].shape}")

    return {"data": data, "labels": labels, "n_valid": n_valid, "times": times,
            "freqs": freqs, "baseline_vals": baseline_vals, "pooled_n": len(pooled_ids)}


# ======================================================================
# 4. SANITY CHECKS
def check_overlap(trials_a, trials_b, labels):
    """A trial that is in both conditions would break the shuffling logic."""
    overlap = set(trials_a.index) & set(trials_b.index)      # trial indices present in both tables
    if overlap:
        raise ValueError(f"{len(overlap)} trials are in BOTH conditions {labels}")
    print("Overlap check OK: no trial belongs to both conditions")


def check_shuffle_logic(data_a, data_b, seed=0):
    """Shuffle the labels once: group sizes and the grand mean must NOT change."""
    n_a, n_b = data_a.shape[0], data_b.shape[0]              # real group sizes
    allx = np.concatenate([data_a, data_b], axis=0)          # all trials together
    perm = np.random.default_rng(seed).permutation(n_a + n_b)   # random order of all trials
    fake_a, fake_b = allx[perm[:n_a]], allx[perm[n_a:]]      # first nA -> fake A, rest -> fake B
    print(f"sizes real  : {n_a}, {n_b}   | shuffled: {fake_a.shape[0]}, {fake_b.shape[0]}")
    print(f"grand mean real: {allx.mean():.6f}")
    print(f"mean A real {data_a.mean():.4f} / B real {data_b.mean():.4f}")
    print(f"mean A fake {fake_a.mean():.4f} / B fake {fake_b.mean():.4f}  (should be closer together)")
    assert fake_a.shape[0] == n_a and fake_b.shape[0] == n_b  # sizes must stay identical


def false_positive_check(data, n_perm=200, pval=0.05, seed=1):
    """
    Split the trials of ONE condition randomly into two halves and test them.
    There is no real effect, so almost nothing should be significant:
      uncorrected ~ a few % of pixels, corrected ~ 0 pixels.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(data.shape[0])                   # random trial order
    half = data.shape[0] // 2                                # split point
    res = permutation_test(data[order[:half]], data[order[half:2 * half]],
                           n_perm=n_perm, pval=pval, seed=seed)
    print(f"False-positive check: uncorrected = {100 * res['sig_uncorrected'].mean():.2f}% of pixels, "
          f"corrected = {100 * res['sig_corrected'].mean():.3f}% of pixels")
    return res


# ======================================================================
# 5. THE PERMUTATION TEST (book 15.3 - 15.6)
def permutation_test(data_a, data_b, n_perm=1000, pval=0.05, seed=0, batch=50):
    """
    data_a: (nA, ...) and data_b: (nB, ...) - first axis = trials, the rest can be anything
            (freqs x times for the map, bands x times for the band lines).
    nA and nB may be DIFFERENT.
    Steps:
      real_diff  = mean(A) - mean(B)
      null       : shuffle the labels of all nA+nB trials n_perm times, keep sizes nA and nB,
                   recompute mean(fakeA) - mean(fakeB) every time
      z_map      = (real_diff - mean(null)) / std(null)         (per pixel, book 15.4/15.5)
      pixel-level threshold   : |z| > z_crit (two-tailed)
      corrected threshold     : min and max z of every shuffled map -> 2.5% / 97.5% tails (book 15.6)
    Returns a dict with all results (shapes = data.shape[1:]).
    """
    n_a, n_b = data_a.shape[0], data_b.shape[0]              # trials per condition
    n_tot = n_a + n_b                                        # all trials
    out_shape = data_a.shape[1:]                             # shape of one map
    X = np.concatenate([data_a, data_b], axis=0).reshape(n_tot, -1).astype(np.float32)
    # X: (trials, pixels) - all trials of both conditions, every map flattened into one row

    # real difference: weights +1/nA for A trials and -1/nB for B trials -> weights @ X = mean(A) - mean(B)
    w_real = np.concatenate([np.full(n_a, 1.0 / n_a), np.full(n_b, -1.0 / n_b)]).astype(np.float32)
    real_diff = (w_real @ X).astype(np.float64)              # (pixels,)

    # random label assignments: one row per permutation (the ORDER OF THE DATA never changes)
    rng = np.random.default_rng(seed)                        # fixed seed -> reproducible
    perms = np.stack([rng.permutation(n_tot) for _ in range(n_perm)])   # (n_perm, n_tot)
    W = np.zeros((n_perm, n_tot), dtype=np.float32)          # weight matrix of all permutations
    rows = np.arange(n_perm)[:, None]
    W[rows, perms[:, :n_a]] = 1.0 / n_a                      # first nA shuffled trials -> "fake A"
    W[rows, perms[:, n_a:]] = -1.0 / n_b                     # remaining nB trials      -> "fake B"

    # --- pass 1: mean and std of the null distribution per pixel ---
    sum_d = np.zeros(X.shape[1])                             # running sum of fake differences
    sum_d2 = np.zeros(X.shape[1])                            # running sum of squares
    for s in range(0, n_perm, batch):                        # batches keep the memory small
        D = (W[s:s + batch] @ X).astype(np.float64)          # (batch, pixels) fake differences
        sum_d += D.sum(axis=0)
        sum_d2 += (D ** 2).sum(axis=0)
    null_mean = sum_d / n_perm                               # mean of null differences
    null_var = (sum_d2 - n_perm * null_mean ** 2) / (n_perm - 1)   # variance (ddof=1)
    null_std = np.sqrt(np.maximum(null_var, 0))              # std of null differences
    null_std[null_std == 0] = np.nan                         # avoid division by zero

    # --- pass 2: most extreme z of every shuffled map (same permutations, same seed) ---
    max_z = np.empty(n_perm)                                 # largest z of each shuffled map
    min_z = np.empty(n_perm)                                 # smallest z of each shuffled map
    for s in range(0, n_perm, batch):
        D = (W[s:s + batch] @ X).astype(np.float64)
        Z = (D - null_mean) / null_std                       # z-map of every shuffled difference
        max_z[s:s + batch] = np.nanmax(Z, axis=1)           # ONE value per map (not per pixel)
        min_z[s:s + batch] = np.nanmin(Z, axis=1)

    z_map = (real_diff - null_mean) / null_std               # real z-map

    z_crit = norm.ppf(1 - pval / 2)                          # pixel-level threshold (two-tailed), e.g. 1.96
    lower = np.percentile(min_z, 100 * pval / 2)             # corrected lower threshold (2.5% tail)
    upper = np.percentile(max_z, 100 - 100 * pval / 2)       # corrected upper threshold (97.5% tail)

    with np.errstate(invalid='ignore'):                      # NaN pixels simply count as "not significant"
        sig_unc = np.abs(z_map) > z_crit                     # significant without multiple-comparison correction
        sig_cor = (z_map < lower) | (z_map > upper)          # significant after extreme-value correction

    return {"real_diff": real_diff.reshape(out_shape),
            "z_map": z_map.reshape(out_shape),
            "null_mean": null_mean.reshape(out_shape),
            "null_std": null_std.reshape(out_shape),
            "sig_uncorrected": sig_unc.reshape(out_shape),
            "sig_corrected": sig_cor.reshape(out_shape),
            "z_crit": z_crit, "lower": lower, "upper": upper,
            "max_z": max_z, "min_z": min_z,
            "mean_a": data_a.mean(axis=0), "mean_b": data_b.mean(axis=0),
            "n_a": n_a, "n_b": n_b, "n_perm": n_perm, "pval": pval}


def band_average_trials(data, freqs, band_ranges, bands=None):
    """
    (trials, freqs, times) -> (trials, n_bands, times): mean over the frequencies of each band,
    separately for every trial. band_ranges = {"theta": (4, 8), ...}
    """
    bands = bands or list(band_ranges.keys())                # which bands, in which order
    out = []
    for band in bands:
        fmin, fmax = band_ranges[band]                       # frequency range of this band
        mask = (freqs >= fmin) & (freqs <= fmax)             # same rule as tfx.get_band_trace
        out.append(data[:, mask, :].mean(axis=1))            # (trials, times)
    return np.stack(out, axis=1), bands


def run_permutation_stats(prep, band_ranges, n_perm=1000, pval=0.05, seed=0, batch=50, bands=None):
    """Runs the test for every event: (1) full time-frequency map, (2) band-average lines."""
    stats = {}
    for ev_name, (data_a, data_b) in prep["data"].items():   # Press, Tone, Release
        print(f"--- {ev_name}: nA={data_a.shape[0]}, nB={data_b.shape[0]}, {n_perm} permutations ---")
        map_res = permutation_test(data_a, data_b, n_perm, pval, seed, batch)       # full TFR map
        band_a, used = band_average_trials(data_a, prep["freqs"], band_ranges, bands)
        band_b, _ = band_average_trials(data_b, prep["freqs"], band_ranges, bands)
        band_res = permutation_test(band_a, band_b, n_perm, pval, seed, batch)      # band lines
        stats[ev_name] = {"map": map_res, "bands": band_res}
    return stats, used


def print_stats_summary(stats):
    """Short text summary: how many pixels/time points are significant per event."""
    for ev, r in stats.items():
        m, b = r["map"], r["bands"]
        print(f"{ev}: MAP  uncorrected {100 * m['sig_uncorrected'].mean():.2f}% | "
              f"corrected {100 * m['sig_corrected'].mean():.3f}% (z thresholds {m['lower']:.2f} / {m['upper']:.2f})")
        print(f"{ev}: BANDS uncorrected {100 * b['sig_uncorrected'].mean():.2f}% | "
              f"corrected {100 * b['sig_corrected'].mean():.3f}% (z thresholds {b['lower']:.2f} / {b['upper']:.2f})")


# ======================================================================
# 6. PLOTS
def _draw_sig_contours(ax, times, freqs, res):
    """Dashed grey line = significant without correction, solid black line = after correction."""
    if res["sig_uncorrected"].any():
        ax.contour(times, freqs, res["sig_uncorrected"].astype(float), levels=[0.5],
                   colors='grey', linewidths=0.8, linestyles='--')
    if res["sig_corrected"].any():
        ax.contour(times, freqs, res["sig_corrected"].astype(float), levels=[0.5],
                   colors='black', linewidths=1.5)


def plot_permutation_maps(stats, times, freqs, event_names, labels, title,
                          freq_range=(1, 99), yticks=None, band_boundaries=None,
                          vmax=None, save_path=None):
    """
    Row 1: difference map (condition A - condition B) with significance contours.
    Row 2: z-map (real difference relative to the null distribution).
    One column per event.
    """
    if yticks is None:
        yticks = [1, 4, 8, 13, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    n_ev = len(event_names)
    if vmax is None:                                         # common colour range for the difference maps
        vmax = max(np.nanmax(np.abs(stats[e]["map"]["real_diff"])) for e in event_names)
    zmax = max(np.nanmax(np.abs(stats[e]["map"]["z_map"])) for e in event_names)   # same for the z-maps

    fig, axes = plt.subplots(2, n_ev, figsize=(6 * n_ev, 9), sharex=True, sharey=True,
                             squeeze=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.12})
    im0 = im1 = None
    for c, ev in enumerate(event_names):
        res = stats[ev]["map"]
        for r, (key, lim) in enumerate([("real_diff", vmax), ("z_map", zmax)]):
            ax = axes[r, c]
            # pcolormesh (instead of imshow) stays correct even if the frequency axis is not evenly spaced
            im = ax.pcolormesh(times, freqs, res[key], cmap='RdBu_r', vmin=-lim, vmax=lim,
                               shading='nearest')
            _draw_sig_contours(ax, times, freqs, res)
            ax.axvline(0, color='k', linestyle='--', linewidth=1.2)
            if band_boundaries:
                for f in band_boundaries:
                    ax.axhline(f, color='grey', linestyle=':', linewidth=0.5)
            ax.set_yticks(yticks)
            ax.set_ylim(freq_range)
            ax.set_xlabel('Time relative to alignment (s)')
            ax.tick_params(axis='x', labelbottom=True)
            ax.tick_params(axis='y', labelleft=True)
            ax.grid(False)
            if r == 0:
                im0 = im
            else:
                im1 = im
        axes[0, c].set_title(f"{ev}  (nA={res['n_a']}, nB={res['n_b']})", fontsize=15, fontweight='bold', pad=10)
    axes[0, 0].set_ylabel(f"Frequency (Hz)\ndifference", fontsize=12)
    axes[1, 0].set_ylabel("Frequency (Hz)\nz-map", fontsize=12)
    fig.colorbar(im0, ax=axes[0, :], pad=0.02, extend='both').set_label(
        f"Power difference (log ratio): {labels[0]}  minus  {labels[1]}")
    fig.colorbar(im1, ax=axes[1, :], pad=0.02, extend='both').set_label("z (vs. permutation null)")
    fig.suptitle(title + "\ngrey dashed = p<0.05 uncorrected | black = extreme-value corrected",
                 fontsize=16, fontweight='bold', y=0.99)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_permutation_bands(stats, times, event_names, bands, labels, title,
                           ylim=None, save_path=None):
    """
    One row per band, one column per event. Two lines = mean log-ratio power of the two conditions.
    Red shading = significant after correction, grey bar at the bottom = significant uncorrected.
    """
    n_b, n_ev = len(bands), len(event_names)
    fig, axes = plt.subplots(n_b, n_ev, figsize=(5.5 * n_ev, 2.6 * n_b), sharex=True,
                             squeeze=False)
    colors = ['tab:blue', 'tab:green']                       # condition A / condition B
    for c, ev in enumerate(event_names):
        res = stats[ev]["bands"]
        for r, band in enumerate(bands):
            ax = axes[r, c]
            ax.plot(times, res["mean_a"][r], color=colors[0], linewidth=1.8,
                    label=f"{labels[0]} (n={res['n_a']})")
            ax.plot(times, res["mean_b"][r], color=colors[1], linewidth=1.8,
                    label=f"{labels[1]} (n={res['n_b']})")
            ax.fill_between(times, 0, 1, where=res["sig_corrected"][r],
                            transform=ax.get_xaxis_transform(), color='red', alpha=0.2, linewidth=0)
            ax.fill_between(times, 0, 0.04, where=res["sig_uncorrected"][r],
                            transform=ax.get_xaxis_transform(), color='grey', linewidth=0)
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
            ax.grid(False)
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
            if ylim:
                ax.set_ylim(ylim)
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_xlabel('Time relative to alignment (s)', fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{bands[r]}\nPower (log ratio)", fontsize=10)
        axes[0, c].set_title(ev, fontsize=15, fontweight='bold', pad=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title + "\nred = significant (corrected) | grey bar = uncorrected",
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
