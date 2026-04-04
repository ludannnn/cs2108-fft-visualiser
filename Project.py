import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks, spectrogram
import io
import time
import math
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="FFT Learning Tool", layout="wide")

# ── Color palette ──────────────────────────────────────────────────────────────
BG       = "#0D1117"   # figure background
AX_BG    = "#161B22"   # axes background
GRID_CLR = "#30363D"   # grid lines
TEXT_CLR = "#E6EDF3"   # labels / ticks
C_CYAN   = "#00D4FF"   # combined signal
C_PINK   = "#FF6B9D"   # frequency spectrum / audio freq
C_GREEN  = "#10B981"   # audio time domain
# Per-component wave colours (cycles if > 5 waves)
WAVE_COLORS = ["#7C3AED", "#FFD60A", "#00D4FF", "#FF6B9D", "#10B981"]

def style_ax(ax, fig, spine_color=None):
    """Apply the dark theme to a matplotlib axes."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.spines[:].set_color(spine_color if spine_color else GRID_CLR)
    if spine_color:
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
    ax.grid(True, color=GRID_CLR, alpha=0.6)

def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)

def draw_decomp_frame(fig, t, y_combined, individual_waves, n_done, cur_idx, cur_progress):
    """Draw one animation frame of the decomposition."""
    num_waves = len(individual_waves)
    nrows = num_waves + 1
    axes = fig.subplots(nrows, 1, sharex=True)
    fig.patch.set_facecolor(BG)

    # ── Residual row (top) ──────────────────────────────────────────────────
    ax0 = axes[0]
    residual = y_combined.copy()
    for j in range(n_done):
        residual -= individual_waves[j]["wave"]
    if cur_idx >= 0 and n_done < num_waves:
        residual -= individual_waves[cur_idx]["wave"] * smoothstep(cur_progress)

    cur_color = WAVE_COLORS[cur_idx % len(WAVE_COLORS)] if cur_idx >= 0 else None
    spine_col = cur_color if (cur_idx >= 0 and n_done < num_waves) else None
    style_ax(ax0, fig, spine_color=spine_col)

    # Ghost of original signal
    ax0.plot(t, y_combined, color=C_CYAN, linewidth=1, alpha=0.15)
    # Residual signal
    ax0.plot(t, residual, color=C_CYAN, linewidth=2)

    # Pulse overlay: the wave currently being extracted
    if cur_idx >= 0 and n_done < num_waves:
        pulse = math.sin(math.pi * cur_progress)          # 0 → 1 → 0 over extraction
        highlight_alpha = pulse * 0.75
        ax0.plot(t, individual_waves[cur_idx]["wave"],
                 color=cur_color, linewidth=1.5, alpha=highlight_alpha)

    ax0.set_ylabel("Residual", color=TEXT_CLR, rotation=0, labelpad=45, va="center", fontsize=8)

    # ── Component rows ──────────────────────────────────────────────────────
    for i in range(num_waves):
        ax = axes[i + 1]
        color = WAVE_COLORS[i % len(WAVE_COLORS)]
        is_active = (i == cur_idx and n_done < num_waves)
        style_ax(ax, fig, spine_color=color if is_active else None)

        label_alpha = 1.0
        if i < n_done:
            ax.plot(t, individual_waves[i]["wave"], color=color, linewidth=1.5)
        elif is_active:
            # Wave grows in during extraction phase
            appear_progress = smoothstep(max(0, (cur_progress - 0.3) / 0.7))
            ax.plot(t, individual_waves[i]["wave"], color=color,
                    linewidth=1.5, alpha=appear_progress)
            ax.set_facecolor("#1A2030")   # subtle active background
            label_alpha = max(0.3, appear_progress)
        else:
            label_alpha = 0.25           # future waves are dimmed

        freq_label = individual_waves[i]["freq"]
        type_label = individual_waves[i]["type"]
        ax.set_ylabel(f"{freq_label}Hz\n{type_label}", rotation=0, labelpad=30,
                      va="center", color=TEXT_CLR, alpha=label_alpha, fontsize=8)

    axes[-1].set_xlabel("Time (seconds)", color=TEXT_CLR)
    fig.tight_layout(h_pad=0.4)

st.title("Signal Processing & FFT Interactive Tool")
st.write("Explore how complex signals are built and deconstructed using the Fast Fourier Transform (FFT).")

# Create tabs for the two different features
tab1, tab2, tab3 = st.tabs(["Manual Wave Builder", "Live Audio Analysis", "Aliasing & Nyquist"])

#manual wave section
with tab1:
    st.header("1. Superposition: Build a Signal")
    st.write("Add multiple sine and cosine waves together to see how they form a complex waveform.")
    
    col_controls, col_plots = st.columns([1, 2])
    
    with col_controls:
        st.subheader("Wave Controls")
        num_waves = st.number_input("Number of components to add:", min_value=1, max_value=5, value=2)
        
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        y_combined = np.zeros_like(t)
        
        # NEW: Create a list to store each individual wave so we can plot them later
        individual_waves = [] 
        
        for i in range(num_waves):
            st.markdown(f"**Component {i+1}**")
            wave_type = st.radio(f"Type", ["Sine", "Cosine"], key=f"type_{i}", horizontal=True)
            freq = st.slider(f"Frequency (Hz)", 1, 50, (i+1)*5, key=f"freq_{i}")
            amp = st.slider(f"Amplitude", 0.1, 10.0, 1.0, key=f"amp_{i}")
            
            if wave_type == "Sine":
                wave = amp * np.sin(2 * np.pi * freq * t)
            else:
                wave = amp * np.cos(2 * np.pi * freq * t)
                
            y_combined += wave
            individual_waves.append({"wave": wave, "freq": freq, "type": wave_type}) # Save it!
            st.divider()

    with col_plots:
        # 1. Plot Time Domain (The Combined Signal)
        st.subheader("Step 1: The Combined Signal (Time Domain)")
        fig_time1, ax_time1 = plt.subplots(figsize=(10, 2.5))
        style_ax(ax_time1, fig_time1)
        ax_time1.plot(t, y_combined, color=C_CYAN, linewidth=2)
        ax_time1.set_xlabel("Time (seconds)")
        ax_time1.set_ylabel("Amplitude")
        st.pyplot(fig_time1)

        # 2. Wave Decomposition — animated
        hdr_col, btn_col = st.columns([3, 1])
        with hdr_col:
            st.subheader("Step 2: Deconstructing the Signal (Intermediary)")
            st.write("The Fourier Transform 'unmixes' the complex signal back into pure components:")
        with btn_col:
            st.write("")   # vertical spacer
            run_anim = st.button("▶ Animate Split", key="btn_animate")

        fig_height = 2.0 * (num_waves + 1)
        decomp_slot = st.empty()

        def draw_static_decomp():
            fig = plt.figure(figsize=(10, fig_height))
            draw_decomp_frame(fig, t, y_combined, individual_waves,
                              n_done=num_waves, cur_idx=-1, cur_progress=1.0)
            with decomp_slot.container():
                st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if run_anim:
            FPWAVE  = 28   # frames to animate each wave being extracted
            HOLD    = 8    # frames to hold at start (combined) and end (fully split)
            FPS     = 20   # target frames per second

            total_frames = HOLD + num_waves * FPWAVE + HOLD

            for frame in range(total_frames):
                adj = frame - HOLD

                if adj < 0:                              # intro hold: show combined
                    n_done, cur_idx, cur_progress = 0, -1, 0.0
                elif adj >= num_waves * FPWAVE:          # end hold: fully decomposed
                    n_done, cur_idx, cur_progress = num_waves, num_waves - 1, 1.0
                else:
                    n_done       = adj // FPWAVE
                    cur_idx      = n_done
                    cur_progress = (adj % FPWAVE) / FPWAVE

                fig = plt.figure(figsize=(10, fig_height))
                draw_decomp_frame(fig, t, y_combined, individual_waves,
                                  n_done, cur_idx, cur_progress)
                with decomp_slot.container():
                    st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                time.sleep(1 / FPS)

        else:
            draw_static_decomp()

        # 3. Plot Frequency Domain (FFT)
        st.subheader("Step 3: The Frequency Spectrum (Fourier Domain)")
        st.write("Finally, we map those individual frequencies onto a spectrum. Each spike represents the presence and amplitude of a component.")
        if st.button("Calculate FFT", key="btn_manual"):
            fft_result = np.fft.fft(y_combined)
            fft_freq = np.fft.fftfreq(len(t), 1/fs)
            magnitudes = np.abs(fft_result) / (fs / 2)
            pos_mask = fft_freq > 0
            
            fig_freq1, ax_freq1 = plt.subplots(figsize=(10, 2.5))
            style_ax(ax_freq1, fig_freq1)
            ax_freq1.stem(fft_freq[pos_mask], magnitudes[pos_mask],
                          linefmt=C_PINK, markerfmt="o", basefmt=GRID_CLR)
            ax_freq1.set_xlim(0, max(60, (num_waves * 10) + 10))
            ax_freq1.set_xlabel("Frequency (Hz)")
            ax_freq1.set_ylabel("Magnitude")
            st.pyplot(fig_freq1)
            


#live audio section
with tab2:
    st.header("Live Audio Analysis")
    st.write("Record a short audio clip. Human voices and environmental sounds contain hundreds of overlapping frequencies.")
    
    st.write("Click the microphone icon to start recording and click again to stop:")
    audio_bytes = audio_recorder(pause_threshold=3.0)
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # Read the audio data
        sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
        
        # Convert stereo to mono if necessary
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
            
        # Create time array
        duration = len(audio_data) / sample_rate
        t_audio = np.linspace(0, duration, len(audio_data), endpoint=False)
        
        # Plot Time Domain of audio
        st.subheader("Time Domain")
        fig_time2, ax_time2 = plt.subplots(figsize=(10, 3))
        style_ax(ax_time2, fig_time2)
        ax_time2.plot(t_audio, audio_data, color=C_GREEN, linewidth=0.5)
        ax_time2.set_xlabel("Time (seconds)")
        ax_time2.set_ylabel("Amplitude")
        st.pyplot(fig_time2)
        

        # Compute and Plot Frequency Domain + Spectrogram
        st.subheader("Frequency Analysis")
        if st.button("Analyze Audio", key="btn_audio"):
            audio_float = audio_data.astype(np.float32)

            # ── FFT with peak annotations ─────────────────────────────────────
            fft_result_audio = np.fft.rfft(audio_float)
            fft_freq_audio   = np.fft.rfftfreq(len(audio_float), 1 / sample_rate)
            magnitudes_audio = np.abs(fft_result_audio) / len(audio_float)

            mask_4k    = fft_freq_audio <= 4000
            mag_in_4k  = magnitudes_audio.copy()
            mag_in_4k[~mask_4k] = 0
            peak_threshold = np.max(mag_in_4k) * 0.05

            peaks_raw, _ = find_peaks(
                magnitudes_audio,
                height=peak_threshold,
                distance=max(1, int(sample_rate / len(audio_float) * 30)),
                prominence=peak_threshold * 0.4,
            )
            # Keep only peaks ≤ 4 kHz, pick top 5 by magnitude
            peaks_4k  = [p for p in peaks_raw if fft_freq_audio[p] <= 4000]
            top_peaks = sorted(peaks_4k, key=lambda i: magnitudes_audio[i], reverse=True)[:5]

            fig_freq2, ax_freq2 = plt.subplots(figsize=(10, 3))
            style_ax(ax_freq2, fig_freq2)
            ax_freq2.plot(fft_freq_audio, magnitudes_audio, color=C_PINK, linewidth=1)
            ax_freq2.set_xlim(0, 4000)
            ax_freq2.set_xlabel("Frequency (Hz)")
            ax_freq2.set_ylabel("Magnitude")

            for pk in top_peaks:
                ax_freq2.annotate(
                    f"{fft_freq_audio[pk]:.0f} Hz",
                    xy=(fft_freq_audio[pk], magnitudes_audio[pk]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7.5, color=TEXT_CLR,
                    arrowprops=dict(arrowstyle="->", color=TEXT_CLR, lw=0.8),
                )
                ax_freq2.plot(fft_freq_audio[pk], magnitudes_audio[pk],
                              "o", color="#FFD60A", markersize=5, zorder=5)

            st.pyplot(fig_freq2)
            plt.close(fig_freq2)

            # ── Spectrogram ───────────────────────────────────────────────────
            st.subheader("Spectrogram")
            st.write("Each vertical slice is an FFT snapshot — this shows how the frequency content changes over time.")

            nperseg = min(512, len(audio_float) // 4)
            f_spec, t_spec, Sxx = spectrogram(
                audio_float, fs=sample_rate,
                nperseg=nperseg, noverlap=nperseg * 3 // 4,
                window="hann",
            )
            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            fig_spec, ax_spec = plt.subplots(figsize=(10, 3.5))
            style_ax(ax_spec, fig_spec)
            pcm = ax_spec.pcolormesh(
                t_spec, f_spec, Sxx_db,
                cmap="plasma", shading="gouraud",
                vmin=np.percentile(Sxx_db, 20),   # clip noise floor
                vmax=Sxx_db.max(),
            )
            ax_spec.set_ylim(0, 4000)
            ax_spec.set_xlabel("Time (seconds)")
            ax_spec.set_ylabel("Frequency (Hz)")
            cbar = fig_spec.colorbar(pcm, ax=ax_spec, pad=0.01)
            cbar.set_label("Power (dB)", color=TEXT_CLR)
            cbar.ax.tick_params(colors=TEXT_CLR)
            st.pyplot(fig_spec)
            plt.close(fig_spec)

# ── Tab 3: Aliasing & Nyquist Explorer ────────────────────────────────────────
with tab3:
    st.header("Aliasing & Nyquist Rate Explorer")
    st.write(
        "The **Nyquist theorem** says you must sample at least **twice** the signal frequency "
        "to capture it faithfully. Sample too slowly and the signal 'folds back' — appearing "
        "at a completely different frequency. Drag the sliders to see it happen."
    )

    ctrl_col, plot_col = st.columns([1, 2])

    with ctrl_col:
        f_signal = st.slider("Signal frequency (Hz)", min_value=1, max_value=300, value=50)
        f_sample = st.slider("Sample rate (Hz)",      min_value=10, max_value=800, value=400)

        nyquist = f_sample / 2

        # Compute aliased frequency
        f_alias_raw = f_signal % f_sample
        if f_alias_raw <= nyquist:
            f_alias    = f_alias_raw
            alias_sign = 1
        else:
            f_alias    = f_sample - f_alias_raw
            alias_sign = -1

        is_aliasing = f_signal > nyquist

        st.divider()
        st.metric("Nyquist rate", f"{nyquist:.0f} Hz", help="= sample rate ÷ 2")
        st.metric("Signal frequency", f"{f_signal} Hz")

        if is_aliasing:
            st.error(f"Aliasing! {f_signal} Hz > Nyquist ({nyquist:.0f} Hz)")
            st.metric("Appears as", f"{f_alias:.1f} Hz", delta=f"{f_alias - f_signal:.1f} Hz")
        else:
            st.success(f"Clean sampling — no aliasing")
            headroom = nyquist - f_signal
            st.metric("Headroom below Nyquist", f"{headroom:.0f} Hz")

    with plot_col:
        # Time window: show 3 full periods of the true signal or 20 sample intervals
        t_end  = max(3.0 / f_signal, 20.0 / f_sample)
        t_cont = np.linspace(0, t_end, 8000)
        t_samp = np.arange(0, t_end, 1.0 / f_sample)
        y_cont = np.sin(2 * np.pi * f_signal * t_cont)
        y_samp = np.sin(2 * np.pi * f_signal * t_samp)
        # Sinc-reconstructed apparent signal from samples (passes through all sample points)
        y_apparent = alias_sign * np.sin(2 * np.pi * f_alias * t_cont)

        fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(9, 6))
        fig.patch.set_facecolor(BG)

        # ── Panel 1: Time domain ──────────────────────────────────────────────
        style_ax(ax_time, fig)
        ax_time.plot(t_cont, y_cont, color=C_CYAN, linewidth=1.5,
                     alpha=0.35 if is_aliasing else 0.8,
                     label=f"True signal ({f_signal} Hz)")

        if is_aliasing:
            ax_time.plot(t_cont, y_apparent, color="#FFD60A", linewidth=1.5,
                         linestyle="--", alpha=0.85,
                         label=f"Apparent signal ({f_alias:.1f} Hz)")

        # Sample stems
        markerline, stemlines, baseline = ax_time.stem(
            t_samp, y_samp, linefmt=C_PINK, markerfmt="o", basefmt=" "
        )
        markerline.set_markersize(5)
        markerline.set_color(C_PINK)
        plt.setp(stemlines, linewidth=0.8, alpha=0.6)
        # Fake legend entry for samples
        ax_time.plot([], [], "o", color=C_PINK, markersize=5, label="Sample points")

        ax_time.set_ylabel("Amplitude", color=TEXT_CLR)
        ax_time.set_xlabel("Time (s)", color=TEXT_CLR)
        leg = ax_time.legend(loc="upper right", fontsize=8,
                             facecolor=AX_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

        # ── Panel 2: Frequency domain — the "folding" view ───────────────────
        style_ax(ax_freq, fig)

        # Shade the aliased zone (beyond Nyquist)
        ax_freq.axvspan(nyquist, f_sample, alpha=0.12, color=C_PINK, label="_nolegend_")
        ax_freq.axvline(nyquist, color="white", linewidth=1, linestyle="--", alpha=0.5,
                        label=f"Nyquist limit ({nyquist:.0f} Hz)")

        ax_freq.vlines(f_signal, 0, 1, color=C_CYAN, linewidth=4,
                       label=f"True: {f_signal} Hz")

        if is_aliasing and f_alias > 0:
            ax_freq.vlines(f_alias, 0, 0.85, color="#FFD60A", linewidth=4,
                           label=f"Appears as: {f_alias:.1f} Hz")
            # Arc arrow showing the fold
            ax_freq.annotate(
                "", xy=(f_alias, 0.75), xytext=(f_signal, 0.75),
                arrowprops=dict(
                    arrowstyle="->", color="#FFD60A", lw=1.5,
                    connectionstyle="arc3,rad=-0.35",
                ),
            )
            ax_freq.text(
                (f_signal + f_alias) / 2, 0.82, "folds back",
                ha="center", fontsize=7.5, color="#FFD60A", alpha=0.8,
            )

        ax_freq.set_xlim(0, f_sample)
        ax_freq.set_ylim(0, 1.15)
        ax_freq.set_xlabel("Frequency (Hz)", color=TEXT_CLR)
        ax_freq.set_ylabel("", color=TEXT_CLR)
        ax_freq.yaxis.set_visible(False)
        ax_freq.text(nyquist + (f_sample - nyquist) * 0.05, 1.05,
                     "aliasing zone", fontsize=7.5, color=C_PINK, alpha=0.7)
        leg2 = ax_freq.legend(loc="upper left", fontsize=8,
                              facecolor=AX_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

        fig.tight_layout(h_pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)