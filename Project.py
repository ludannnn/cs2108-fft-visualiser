import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
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
tab1, tab2 = st.tabs(["Manual Wave Builder", "Live Audio Analysis"])

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
        

        # Compute and Plot Frequency Domain (Audio)
        st.subheader("Frequency Domain")
        if st.button("Analyze Audio Frequencies", key="btn_audio"):
            # Use rfft (Real FFT) which is optimized for real-world data like audio
            fft_result_audio = np.fft.rfft(audio_data)
            fft_freq_audio = np.fft.rfftfreq(len(t_audio), 1/sample_rate)
            magnitudes_audio = np.abs(fft_result_audio) / len(t_audio)
            
            fig_freq2, ax_freq2 = plt.subplots(figsize=(10, 3))
            style_ax(ax_freq2, fig_freq2)
            ax_freq2.plot(fft_freq_audio, magnitudes_audio, color=C_PINK)
            ax_freq2.set_xlim(0, 4000)
            ax_freq2.set_xlabel("Frequency (Hz)")
            ax_freq2.set_ylabel("Magnitude")
            st.pyplot(fig_freq2)