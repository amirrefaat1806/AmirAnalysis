'''
tfr_functions.py

'''

import numpy as np # numbers and arrays
import pandas as pd #table
import mne #tfr
import matplotlib.pyplot as plt #plots
import spikeinterface.full as si #Neuropixels
from matplotlib.patches import Polygon







# 1. Loading 
def load_recording(path):
    """Load the (downsampled) recording."""
    return si.load(path)


def load_behavior(session_dir, csv_rel_path, outcome, trial_type):
    """
    Load trials.csv, build the tone_np_s column, and filter trials by
    outcome/trial_type. Returns (full_behavior_df, filtered_trials_df).
    """
    beh = pd.read_csv(str(session_dir) + csv_rel_path)

    beh["tone_np_s"] = (beh["tone"] - beh["press"]) + beh["press_np_s"]

    filtered = beh[(beh[outcome] == 1) & (beh[trial_type] == 1)]

    print(f"Total trials: {len(beh)}")
    print(f"{outcome} trials: {(beh[outcome] == 1).sum()}")
    print(f"{trial_type} trials: {(beh[trial_type] == 1).sum()}")
    print(f"{trial_type} AND {outcome} trials: {len(filtered)}")

    return beh, filtered
# ----------------------------------------------------------------------






# 2. EXTRACTION + PER-BAND TFR (raw, no baseline applied yet)
def extract_raw_power(ds_data, trials_df, channel_id, events, band_settings,
                       tmin, tmax, fs):
    """
    For one channel: extract the trial windows around every event in
    `events` (a dict {event_name: behavior_column}), and compute the
    per-band Morlet TFR (not yet baseline-corrected).

    Returns: dict {event_name: [power_band_0, power_band_1, ...]}
    where each power_band_i is an MNE AverageTFR (per band).
    """
    n_pre = round(fs * abs(tmin)) #how many samples to grab before the event
    n_post = round(fs * tmax) #how many samples to grab after the event
    n_total = n_pre + n_post #how many total samples to grab for the event

    raw_power = {} #Creates an empty dictionary. This will end up holding one entry per event (Press, Tone, Release)

    for name, column in events.items():
        
        align_idx = (trials_df[column].dropna().values * fs).astype(int)
        #Takes the event-time column (in seconds), drops any missing values (dropna), 
        # converts it to a plain NumPy array (.values), multiplies by the sampling rate to convert seconds → sample numbers,
        # then converts to whole numbers (astype(int)).


        arr_list = []  # An empty list that will collect the extracted signal snippet for each valid trial

        for idx in align_idx: #Loops over every event sample-index we just computed (one per trial)
            
            start, end = idx - n_pre, idx + n_post #Calculates the start and end sample of the extraction window around this particular event.
           
            trace = ds_data.get_traces(start_frame=start, end_frame=end,channel_ids=[channel_id],return_in_uV=True)
            #Pulls the actual signal from the recording, for this one channel, from start to end, in microvolts.


            if trace.shape[0] == n_total: # Only keeps this trial if the extracted piece has exactly the expected length 
                                          # If good, adds the 1D signal (trace[:, 0]) to the list.
                arr_list.append(trace[:, 0]) 

        arr_3 = np.array(arr_list)[:, np.newaxis, :]
        #Turns the list of trials into a 3D array shaped (trials, channels, time). 
        # np.newaxis inserts the "channel" dimension in the middle (since MNE expects 3D data even for one channel).

        info = mne.create_info(ch_names=[channel_id], sfreq=fs, ch_types='seeg')
        #Builds an MNE "info" object — basically metadata describing the channel name, sampling rate, and channel type.

        epochs = mne.EpochsArray(arr_3, info, tmin=tmin, verbose=False)
        #Wraps our raw array + metadata into an MNE "Epochs" object (MNE's standard container for trial-based data), 
        # telling it that time 0 in each trial is tmin seconds after the start of the window. 
        # verbose=False just silences MNE's status messages.

        
        band_power_objs = [] #Empty list to collect the TFR result for each frequency band.

        for band in band_settings: #Loops through your list of band definitions (delta, theta, alpha, beta, gamma).

            power = epochs.compute_tfr(method='morlet', freqs=band['freqs'], n_cycles=band['n_cycles'],use_fft=True, return_itc=False, average=True,decim=1, n_jobs=1, verbose=False,)
            #Runs the actual time-frequency computation using Morlet wavelets, for this band's frequencies and n_cycles setting. 
            # average=True means it averages across trials for you. use_fft=True is a speed setting. 
            # return_itc=False means we don't also compute inter-trial coherence. decim=1 means don't downsample the time axis.
            # n_jobs=1 means use 1 CPU core.
            
            band_power_objs.append(power) #Saves this band's power result into the list.

        raw_power[name] = band_power_objs #Stores the whole list of band results under this event's name
        print(f"{name}: {len(arr_list)} valid trials, "
              f"{[p.data.shape for p in band_power_objs]}")

    return raw_power
# ----------------------------------------------------------------------






# 3.1 BASELINE CORRECTION (own)
def compute_own_baseline(raw_power, band_settings, baseline_window, crop_window): 
    #Defines the function for the "own baseline" method — each event corrected against its own pre-event period.
    """
    Each event is baselined against its OWN pre-event window
    ("Own Baseline"). Returns (results_dict, times, freqs).
    """
    names = list(raw_power.keys()) #Gets the list of event names (["Press", "Tone", "Release"]) from the dictionary.

    results = {name: [] for name in names} #Creates a new dictionary where every event name starts with an empty list — will hold the corrected band data.
    
    times = None #A placeholder variable, will be filled in later with the actual time axis.

    for b_idx in range(len(band_settings)):
    #Loops through the band index (0, 1, 2, 3, 4 — one per frequency band)

        for name in names: #For that band, loops through every event.

            power = raw_power[name][b_idx].copy() #Grabs this event's TFR result for this band, and makes a copy
            #important, so we don't accidentally modify the original data (since the next function needs the untouched original).

            power.apply_baseline(baseline=baseline_window, mode='logratio')
            #Applies baseline correction: divides the whole signal by the average power in the baseline window, then takes the log
            # this makes it a "log ratio vs. baseline".

            power.crop(tmin=crop_window[0], tmax=crop_window[1]) 
            #Cuts the data down to just the time window you actually want to look at/plot.
            
            results[name].append(power.data[0]) 
            #Extracts the actual numeric data (power.data[0] picks the one channel we have) and adds it to that event's list.
            
            times = power.times #Saves the time axis (same every loop, since the crop window is identical each time — just keeps overwriting with the same values).

    for name in names:
        results[name] = np.concatenate(results[name], axis=0)
    #After the loops finish, for each event: stacks together the 5 separate band arrays (delta, theta, alpha, beta, gamma) 
    # into one single array — stitching the frequency axis together.

    freqs = np.concatenate([band['freqs'] for band in band_settings])
    #Also stitches together the actual frequency values from all 5 bands into one array.

    return results, times, freqs
    #Returns the three things: the corrected data per event, the time axis, the frequency axis.


# 3.2 BASELINE CORRECTION (press)
def compute_event_baseline(raw_power, band_settings, baseline_window,
                            crop_window, baseline_event):
    """
    Every event is baselined against ONE reference event's pre-event
    window ("Press Baseline"). Returns (results_dict, times, freqs).
    """
    names = list(raw_power.keys())
    results = {name: [] for name in names}
    times = None #same thing

    for b_idx, band in enumerate(band_settings):
        #Loops through bands, this time also grabbing the band's index number automatically via enumerate.


        ref_power = raw_power[baseline_event][b_idx] #Gets the reference event's (Press) TFR data for this band
        #this is what we'll measure the baseline from

        ref_times = ref_power.times #Gets that reference event's time axis.

        b_mask = (ref_times >= baseline_window[0]) & (ref_times <= baseline_window[1])
        #Builds a "mask" (a list of True/False) marking which time points fall inside your baseline window (e.g. -0.5 to -0.2 s)

        baseline_val = ref_power.data[:, :, b_mask].mean(axis=-1, keepdims=True)
        #Selects only the baseline-window time points from the reference event's data, then averages across that time window 
        #giving one baseline power value per frequency.

        crop_mask = (ref_times >= crop_window[0]) & (ref_times <= crop_window[1])
        # Same idea as b_mask, but marking which time points fall inside the window you want to keep for plotting.


        for name in names: #Now loop through every event (Press, Tone, Release) to apply this baseline to each of them.

            power = raw_power[name][b_idx] #Gets this event's raw (uncorrected) TFR data for this band.

            band_result = np.log10(power.data / baseline_val)[0] #Divides this event's power by the baseline value, then takes log base 10 
            #this is the "log ratio vs. baseline" correction, computed manually. [0] picks out the single channel.

            results[name].append(band_result[:, crop_mask]) #Crops down to your time window of interest, and adds it to this event's list.

        times = ref_times[crop_mask] #Saves the cropped time axis

    for name in names:
        results[name] = np.concatenate(results[name], axis=0)
    freqs = np.concatenate([band['freqs'] for band in band_settings])
    return results, times, freqs #Same stitching and return as in the previous function.
# ----------------------------------------------------------------------


# 4. PLOTTING - single-channel compare (own baseline vs event baseline)
#    Used for BOTH the "original" (full-range) and "zoomed" plot -
#    just call it twice with a different freq_range/yticks.
def plot_tfr_compare(results_own, results_baseline, times, freqs, event_names,
                      baseline_event_name, title, freq_range=(1, 99),
                      yticks=None, band_boundaries=None, save_path=None,vmax=None):
    #Defines the plotting function. Takes the two result dictionaries (own-baseline and event-baseline), 
    #the time/frequency axes, the list of event names, which event was used as baseline (for the subtitle), 
    #the title text, and some optional styling settings with defaults.
    """
    Two-column figure: left = own baseline, right = <baseline_event_name> baseline.
    One row per event in event_names.
    """
    if yticks is None:
        yticks = [1, 4, 8, 13, 20, 30, 40, 50, 60, 70, 80, 90, 99] #If you didn't specify custom y-axis tick marks, use this default list.

    n = len(event_names) #Counts how many events there are (usually 3) — this decides how many rows the figure needs.

    if vmax is None:
        all_vals = np.concatenate([results_own[k].ravel() for k in event_names] +[results_baseline[k].ravel() for k in event_names])
        vmax = np.nanmax(np.abs(all_vals))

    fig, axes = plt.subplots(n, 2, figsize=(18, 6 * n), sharex=True, sharey=True,gridspec_kw={'hspace': 0.35, 'wspace': 0.15})
    #Creates the figure: n rows, 2 columns of subplots (own-baseline vs event-baseline), sized based on how many events there are, 
    # sharing x/y axes between subplots, with some spacing settings.

    if n == 1:
        axes = axes[np.newaxis, :] 
    #if there's only 1 event, axes comes back as a 1D array instead of 2D, 
    # so this reshapes it so the rest of the code (which expects 2D indexing like axes[row, col]) still works.

    im = None # Placeholder for the image object (needed later for the colorbar).

    for row, name in enumerate(event_names): #Loops through each event, keeping track of the row number.

        for col, results in enumerate([results_own, results_baseline]): #For each event, loops through the two result dictionaries 
            #column 0 = own baseline, column 1 = event baseline.

            ax = axes[row, col] #Picks out the specific subplot (row, column) we're about to draw into.

            im = ax.imshow(results[name], aspect='auto', origin='lower',extent=[times[0], times[-1], freqs[0], freqs[-1]],cmap='viridis', vmin=-vmax, vmax=vmax)
            #Draws the actual heatmap image: the data, stretched to fit (aspect='auto'), with low frequencies at the bottom (origin='lower')
            #the axes labeled by actual time/frequency values (extent), using the viridis color scheme, with our computed color limits.

            ax.axvline(0, color='white', linestyle='--', linewidth=1.5) #Draws a white dashed vertical line at time = 0 (the event itself)


            if band_boundaries:
                for f in band_boundaries:
                    ax.axhline(f, color='white', linestyle='--', linewidth=0.5)
                    #If you gave frequency-band boundary lines (like 4, 8, 13, 30 Hz), draws a thin white dashed horizontal line at each one.

            ax.set_yticks(yticks)
            ax.set_ylim(freq_range)
            #Sets which frequency values get tick labels, and limits the y-axis to your chosen range (full or zoomed).

            ax.set_xlabel('Time relative to alignment (s)')
            ax.tick_params(axis='x', labelbottom=True)
            ax.tick_params(axis='y', labelleft=True)
            #Labels the x-axis, and forces both x and y tick labels to show on every subplot 
            # (normally sharex/sharey would hide some of them).

        axes[row, 0].set_ylabel('Frequency (Hz)')
        # After both columns for this row are drawn, adds a y-axis label to the left subplot only.

        axes[row, 0].text(-0.18, 0.5, name, transform=axes[row, 0].transAxes,rotation=90, va='center', ha='center',fontsize=14, fontweight='bold')
        #Adds rotated text to the left of the row (outside the plot area) showing the event name (Press/Tone/Release) as a row label.
    
    axes[0, 0].set_title('Own Baseline', fontsize=16, fontweight='bold', pad=12)
    axes[0, 1].set_title(f'pre-{baseline_event_name} Baseline', fontsize=16,fontweight='bold', pad=12)
    #After all rows are done, adds column titles only to the top row: "Own Baseline" and e.g. "Press Baseline".

    #Adds the big overall title at the top of the whole figure (this is the text you built with tfx.make_title(...)).
    fig.suptitle(title, fontsize=20, fontweight='bold', x=0.45, y=0.98)

    cbar = fig.colorbar(im, ax=axes, pad=0.02, extend='both')
    cbar.set_label('Power (log ratio vs. baseline)', fontsize=12)
    #Adds one shared colorbar for the whole figure, with a label.

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
        #If you gave a save path (i.e. CONFIG["save"] = True)
        # saves the figure as a PNG at high resolution, tightly cropped, and prints a confirmation.

    plt.show() 
# ----------------------------------------------------------------------



# 5. ACROSS-CHANNELS
def extract_and_baseline_multichannel(ds_data, trials_df, channel_ids, events,
                                       band_settings, tmin, tmax, fs,
                                       baseline_window, crop_window,
                                       baseline_event):
    #Defines the function that repeats the whole extraction+baseline process for a list of channels (instead of just one)
    """
    Runs extract_raw_power + compute_event_baseline for every channel in
    channel_ids. Returns (all_results {channel_id: results_dict}, times, freqs).
    """
    all_results = {} #Empty dictionary to collect results per channel; placeholders for time/frequency axes
    times = freqs = None

    for ch in channel_ids: #Loops through every channel ID you give it

        raw_power = extract_raw_power(ds_data, trials_df, ch, events, band_settings, tmin, tmax, fs)
        #Calls the extraction function (the one explained above) for this one channel

        results, times, freqs = compute_event_baseline(raw_power, band_settings, baseline_window, crop_window, baseline_event)
        #Applies the event-baseline correction to this channel's data

        all_results[ch] = results
        #Stores this channel's corrected results in the big dictionary, keyed by channel ID.

        print(f"Done: {ch}") #Prints progress, so you can see it working through channels one by one

    return all_results, times, freqs #Returns everything once all channels are done.


def get_probe_df(ds_data): #Defines a function to get the probe's physical layout (needed for the mini probe-location plot)
    """Probe geometry + channel_id, for the probe-location mini-plot."""
    probe = ds_data.get_probe() #Asks the recording object for its probe geometry information.
    probe_df = probe.to_dataframe().copy() # Converts that probe info into a table (DataFrame), and makes a safe copy.
    channel_ids = ds_data.get_channel_ids() #Gets the list of all channel IDs in the recording.
    probe_df['channel_id'] = [channel_ids[i] for i in probe.device_channel_indices]
    #Adds a new column to the probe table matching each row to its actual channel ID name 
    #(since the probe's internal indices don't automatically carry the channel names).
    return probe_df #Returns the completed probe table.

def select_every_nth_channel(probe_df, n):
    """Pick every nth channel from the probe table (n=20 -> every 20th)."""
    return probe_df.iloc[::n].reset_index(drop=True)

def plot_across_channels(all_results, times, freqs, event_names, selected_df,
                          probe_df, baseline_event_name, title,
                          freq_range=(1, 99), yticks=None,
                          band_boundaries=None, save_path=None, vmax=None, contour=None): #Defines the big multi-channel plotting function.
    
    """
    One row per channel (in selected_df), one column per event, plus a
    probe-location mini-plot in the last column.
    selected_df needs columns: channel_id, x, y.
    """
    if yticks is None:
        yticks = [1, 4, 8, 13, 20, 30, 40, 50, 60, 70, 80, 90, 99] #Default y-axis ticks if you didn't give custom ones.

    n_channels = len(selected_df) #Counts how many channels you're plotting — this decides the number of rows.

    if vmax is None:
        all_values = np.concatenate([
            all_results[ch][cond].ravel()[np.isfinite(all_results[ch][cond].ravel())]
            for ch in all_results for cond in event_names
        ])
        vmax = np.nanmax(np.abs(all_values))

    n_cols = len(event_names) + 1  #Number of columns = one per event, plus one extra for the probe-location mini-plot.

    fig, axes = plt.subplots(n_channels, n_cols, figsize=(5.5 * n_cols, 3.2 * n_channels),gridspec_kw={'width_ratios': [1] * len(event_names) + [0.8],'wspace': 0.18, 'hspace': 0.45})
    #Creates the grid of subplots: one row per channel, columns as described, 
    # with the last column narrower (0.45) since it's just a small probe diagram.

    if n_channels == 1: 
        axes = axes[np.newaxis, :] #Same fix as before — if there's only 1 channel, reshape so 2D indexing still works

    im = None #Placeholder for the colorbar image later.

    for row, (_, channel_row) in enumerate(selected_df.iterrows()):
        #Loops through your selected_df table, row by row — channel_row holds that row's data (channel_id, x, y), 
        # row is the row number for indexing into axes.

        channel_id = channel_row['channel_id']
        channel_x, channel_y = channel_row['x'], channel_row['y']
        #Pulls out this channel's ID and its x/y position on the probe.

        for col, name in enumerate(event_names): #Loops through each event, for this channel's row.
            ax = axes[row, col] #Picks the subplot to draw into.

            im = ax.imshow(all_results[channel_id][name], aspect='auto', origin='lower',extent=[times[0], times[-1], freqs[0], freqs[-1]],cmap='viridis', vmin=-vmax, vmax=vmax, interpolation='bilinear')
              # Draws this channel/event's TFR heatmap, same as before but with interpolation='bilinear' 
              # added (smooths the image slightly, since these subplots are smaller).

            ax.axvline(0, color='white', linestyle='--', linewidth=1.2)#Dashed vertical line at time 0.
          
            if band_boundaries:
                for f in band_boundaries:
                    ax.axhline(f, color='white', linestyle='--', linewidth=0.6) #Optional frequency-band boundary lines.

            ax.set_ylim(freq_range)
            ax.set_yticks(yticks)
            ax.set_xlim(times[0], times[-1])
            #Sets the visible frequency and time range and ticks

            ax.grid(False) #Turns off any background grid lines.

            ax.tick_params(axis='both', which='both', length=3, labelsize=7)
            #Makes the tick marks small (since these subplots are small too).

            ax.set_xlabel('Time (s)', fontsize=8)
            #Small x-axis label.

            if col == 0:
                ax.set_ylabel('Frequency (Hz)', fontsize=8)
                ax.text(-0.22, 0.5, channel_id, transform=ax.transAxes, rotation=90,
                         va='center', ha='center', fontsize=14, fontweight='bold')
            else:
                ax.tick_params(axis='y', labelleft=False)
            #Only the first column gets a y-axis label and the channel-name text on the side; other columns hide their y-tick labels to avoid clutter

        ax_probe = axes[row, -1]

        if contour is not None:
            ax_probe.add_patch(Polygon(contour, closed=True, facecolor='#f2f2f2', edgecolor='#888888', linewidth=1.0, zorder=0))

        ax_probe.scatter(probe_df['x'], probe_df['y'], s=4, marker='s', color='#b0b0b0', edgecolors='none', zorder=2)

        ax_probe.scatter(channel_x, channel_y, s=20, color='red', edgecolors='black', linewidths=1, zorder=10)

        

        ax_probe.set_xlim(probe_df['x'].min() - 100, probe_df['x'].max() + 700)
        ax_probe.set_ylim(probe_df['y'].min() - 150, probe_df['y'].max() + 150)
        ax_probe.tick_params(labelsize=6)
        for side in ['top', 'right']:
            ax_probe.spines[side].set_visible(False)
        ax_probe.grid(False)

        for spine in ax_probe.spines.values():
            spine.set_visible(False)
            #Removes all axis ticks, grid lines, and the box border around this mini plot
            # makes it look like a clean diagram rather than a normal chart.

    for col, name in enumerate(event_names):
        axes[0, col].set_title(name, fontsize=15, fontweight='bold', pad=10)
    axes[0, -1].set_title('Probe Location', fontsize=13, fontweight='bold', pad=10)
#After all rows are drawn, adds column titles to just the top row (event names + "Probe Location").

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.998)
    #Adds the overall figure title.

    plt.subplots_adjust(left=0.08, right=0.93, top=0.985, bottom=0.025, wspace=0.15, hspace=0.45)
    #Fine-tunes the spacing/margins around the whole figure.

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()

# ----------------------------------------------------------------------



# 6. TITLE / FILENAME HELPERS - build everything from CONFIG 
def make_title(config, extra=""):
    #Defines a helper that builds a plot title automatically from your CONFIG. extra is optional additional text (like "(zoomed)").
    n = config.get("n_trials")
    n_txt = f", n={n}" if n is not None else ""
    base = (f"TFR — {config['outcome']} / {config['trial_type']} trials "
            f"({config.get('analysis_label', '')} {n_txt})")
    #Builds the main title text using an f-string, pulling outcome, trial_type, 
    # and analysis_label straight out of your CONFIG dictionary. .get(...) is used for analysis_label 
    #so it won't crash even if that key is missing.

    return f"{base} {extra}".strip() 
#Adds the extra text (if any) onto the end, and .strip() removes any leftover blank space if extra was empty.


def make_save_path(config, suffix): #Defines a helper that builds the full file path for saving a PNG, or returns nothing if saving is off.
    
    save_dir = config.get("save_dir") #Reads the save_dir value from CONFIG (or None if missing).

    if not save_dir or not config.get("save", False):
        return None
    #If there's no save folder specified, OR CONFIG["save"] is False, this function returns None — meaning "don't save".
    
    fname = f"{config['outcome']}_{config['trial_type']}_{suffix}.png"
    #Builds an automatic filename from your outcome, trial type, and whatever suffix was passed in (e.g. "imec0.ap#AP0_full").

    return str(save_dir) + "\\" + fname
#Joins the folder and filename together with a Windows-style backslash, and returns the complete path.

# 7. COMMON (POOLED) BASELINE
def load_pooled_trials(session_dir, csv_rel_path, conditions=None):
    beh = pd.read_csv(str(session_dir) + csv_rel_path)
    beh["tone_np_s"] = (beh["tone"] - beh["press"]) + beh["press_np_s"]
    if conditions is None:
        pooled = beh.dropna(subset=["press_np_s"])
    else:
        parts = [beh[(beh[o] == 1) & (beh[t] == 1)] for o, t in conditions]
        pooled = pd.concat(parts)
        pooled = pooled[~pooled.index.duplicated()]
    print(f"Pooled baseline trials: {len(pooled)}")
    return pooled


def get_baseline_values(ref_power_list, baseline_window):
    """ref_power_list = raw_power[baseline_event] (one AverageTFR per band).
    Returns a list (one per band) with the mean baseline power per frequency."""
    vals = []
    for ref_power in ref_power_list:
        m = (ref_power.times >= baseline_window[0]) & (ref_power.times <= baseline_window[1])
        vals.append(ref_power.data[:, :, m].mean(axis=-1, keepdims=True))
    return vals


def compute_common_baseline(raw_power, band_settings, baseline_vals, crop_window):
    """Like compute_event_baseline, but the baseline values come from outside
    (pooled trials). Returns (results, times, freqs)."""
    names = list(raw_power.keys())
    results = {name: [] for name in names}
    times = None
    for b_idx in range(len(band_settings)):
        for name in names:
            power = raw_power[name][b_idx]
            crop_mask = (power.times >= crop_window[0]) & (power.times <= crop_window[1])
            res = np.log10(power.data / baseline_vals[b_idx])[0]
            results[name].append(res[:, crop_mask])
            times = power.times[crop_mask]
    for name in names:
        results[name] = np.concatenate(results[name], axis=0)
    freqs = np.concatenate([b['freqs'] for b in band_settings])
    return results, times, freqs


# ======================================================================
# 8. BAND AVERAGE TRACE (one line = mean power over a frequency range)
def get_band_trace(channel_results, freqs, fmin, fmax):
    """Average over frequencies fmin..fmax -> dict {event: 1D array over time}."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    return {ev: data[mask].mean(axis=0) for ev, data in channel_results.items()}


def plot_band_average(results_by_label, times, freqs, event_names, band, band_ranges,
                      channel_id, ylim=None, save_path=None):
    """
    results_by_label = {"correct short": all_results_A[ch], "late short": all_results_B[ch]}
    (a single entry is fine too). One panel per event, one line per condition.
    """
    fmin, fmax = band_ranges[band]
    fig, axes = plt.subplots(1, len(event_names), figsize=(5 * len(event_names), 3.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for label, ch_res in results_by_label.items():
        trace = get_band_trace(ch_res, freqs, fmin, fmax)
        for ax, ev in zip(axes, event_names):
            ax.plot(times, trace[ev], label=label, linewidth=1.8)
    for ax, ev in zip(axes, event_names):
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title(ev, fontweight='bold')
        ax.set_xlabel('Time relative to alignment (s)')
        ax.grid(False)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        if ylim:
            ax.set_ylim(ylim)
    axes[0].set_ylabel('Power(log ratio)')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{band} ({fmin}-{fmax} Hz) — {channel_id}", fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ======================================================================
# 9. TWO-CONDITION COMPARISON WITH ONE SHARED (POOLED) PRE-PRESS BASELINE
def make_save_path_named(config, name):
    save_dir = config.get("save_dir")
    if not save_dir or not config.get("save", False):
        return None
    return str(save_dir) + "\\" + name + ".png"


def compute_conditions_common_baseline(ds_data, session_dir, csv_rel_path, conditions,
                                       channel_id, events, band_settings, tmin, tmax, fs,
                                       baseline_window, crop_window, baseline_event):
    """
    conditions = [("correct", "long"), ("late", "long")]
    1) pool the trials of ALL listed conditions
    2) compute ONE pre-event baseline from the pooled trials (baseline_event only)
    3) apply that same baseline to each condition separately
    Returns (results_by_cond, n_by_cond, pooled_df, times, freqs).
    """
    pooled_df = load_pooled_trials(session_dir, csv_rel_path, conditions)
    ref_power = extract_raw_power(ds_data, pooled_df, channel_id,
                                  {baseline_event: events[baseline_event]},
                                  band_settings, tmin, tmax, fs)
    baseline_vals = get_baseline_values(ref_power[baseline_event], baseline_window)

    results_by_cond, n_by_cond = {}, {}
    times = freqs = None
    for outcome, trial_type in conditions:
        _, trials = load_behavior(session_dir, csv_rel_path, outcome, trial_type)
        label = f"{outcome} / {trial_type}"
        raw = extract_raw_power(ds_data, trials, channel_id, events,
                                band_settings, tmin, tmax, fs)
        res, times, freqs = compute_common_baseline(raw, band_settings,
                                                    baseline_vals, crop_window)
        results_by_cond[label] = res
        n_by_cond[label] = len(trials)
    return results_by_cond, n_by_cond, pooled_df, times, freqs


def plot_conditions_compare(results_by_cond, n_by_cond, times, freqs, event_names,
                            title, freq_range=(1, 99), yticks=None,
                            band_boundaries=None, vmax=0.5, save_path=None):
    """One row per condition, one column per event, one shared colorbar."""
    if yticks is None:
        yticks = [1, 4, 8, 13, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    conds = list(results_by_cond.keys())
    n_rows, n_cols = len(conds), len(event_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             sharex=True, sharey=True, squeeze=False,
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.15})
    im = None
    for r, cond in enumerate(conds):
        for c, ev in enumerate(event_names):
            ax = axes[r, c]
            im = ax.imshow(results_by_cond[cond][ev], aspect='auto', origin='lower',
                           extent=[times[0], times[-1], freqs[0], freqs[-1]],
                           cmap='viridis', vmin=-vmax, vmax=vmax)
            ax.axvline(0, color='white', linestyle='--', linewidth=1.5)
            if band_boundaries:
                for f in band_boundaries:
                    ax.axhline(f, color='white', linestyle='--', linewidth=0.5)
            ax.set_yticks(yticks)
            ax.set_ylim(freq_range)
            ax.set_xlabel('Time relative to alignment (s)')
            ax.tick_params(axis='x', labelbottom=True)
            ax.tick_params(axis='y', labelleft=True)
            ax.grid(False)
        axes[r, 0].set_ylabel('Frequency (Hz)')
        axes[r, 0].text(-0.22, 0.5, f"{cond} (n={n_by_cond[cond]})",
                        transform=axes[r, 0].transAxes, rotation=90,
                        va='center', ha='center', fontsize=13, fontweight='bold')
    for c, ev in enumerate(event_names):
        axes[0, c].set_title(ev, fontsize=16, fontweight='bold', pad=12)

    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    cbar = fig.colorbar(im, ax=axes, pad=0.02, extend='both')
    cbar.set_label('Power (log ratio vs. pooled pre-Press baseline)', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

# ======================================================================
# 10. BAND LINES PER CONDITION (same layout as plot_conditions_compare, no heatmap)
BAND_COLORS = {"delta": "#1f77b4", "theta": "#2ca02c", "alpha": "#ff7f0e",
               "beta": "#d62728", "gamma": "#9467bd"}


def plot_conditions_band_lines(results_by_cond, n_by_cond, times, freqs, event_names,
                               band_ranges, title, bands=None, ylim=None,
                               save_path=None):
    """
    One row per condition, one column per event. In every panel: one line per
    band (mean log-ratio power over that band's frequency range), each band in
    its own color, one shared legend at the top.
    """
    bands = bands or list(band_ranges.keys())
    conds = list(results_by_cond.keys())
    n_rows, n_cols = len(conds), len(event_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    for r, cond in enumerate(conds):
        for band in bands:
            fmin, fmax = band_ranges[band]
            trace = get_band_trace(results_by_cond[cond], freqs, fmin, fmax)
            for c, ev in enumerate(event_names):
                axes[r, c].plot(times, trace[ev], color=BAND_COLORS.get(band),
                                linewidth=1.8, label=f"{band} ({fmin}-{fmax} Hz)")
        for c, ev in enumerate(event_names):
            ax = axes[r, c]
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
            ax.set_xlabel('Time relative to alignment (s)')
            ax.tick_params(axis='x', labelbottom=True)
            ax.tick_params(axis='y', labelleft=True)
            ax.grid(False)
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
            if ylim:
                ax.set_ylim(ylim)
        axes[r, 0].set_ylabel('Power (log ratio)')
        axes[r, 0].text(-0.25, 0.5, f"{cond} (n={n_by_cond[cond]})",
                        transform=axes[r, 0].transAxes, rotation=90,
                        va='center', ha='center', fontsize=13, fontweight='bold')

    for c, ev in enumerate(event_names):
        axes[0, c].set_title(ev, fontsize=16, fontweight='bold', pad=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.0)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               ncol=len(bands), frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.subplots_adjust(hspace=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()