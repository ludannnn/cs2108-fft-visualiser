import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import io
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="FFT Learning Tool", layout="wide")
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
        ax_time1.plot(t, y_combined, color='#2E86C1', linewidth=2)
        ax_time1.set_xlabel("Time (seconds)")
        ax_time1.set_ylabel("Amplitude")
        ax_time1.grid(True, alpha=0.3)
        st.pyplot(fig_time1)

        # 2. THE NEW INTERMEDIARY STEP: Wave Decomposition
        st.subheader("Step 2: Deconstructing the Signal (Intermediary)")
        st.write("The Fourier Transform essentially 'unmixes' the complex signal back into these pure components:")
        
        # Create a dynamic number of subplots based on how many waves the user made
        fig_inter, axs = plt.subplots(num_waves, 1, figsize=(10, 1.5 * num_waves), sharex=True)
        
        # If there's only 1 wave, axs is not an array, so we make it one to prevent errors
        if num_waves == 1: axs = [axs] 
        
        for i, wave_data in enumerate(individual_waves):
            axs[i].plot(t, wave_data["wave"], color='#E67E22')
            axs[i].set_ylabel(f"{wave_data['freq']}Hz\n{wave_data['type']}", rotation=0, labelpad=30, va='center')
            axs[i].grid(True, alpha=0.3)
            
        axs[-1].set_xlabel("Time (seconds)")
        plt.tight_layout()
        st.pyplot(fig_inter)

        # 3. Plot Frequency Domain (FFT)
        st.subheader("Step 3: The Frequency Spectrum (Fourier Domain)")
        st.write("Finally, we map those individual frequencies onto a spectrum. Each spike represents the presence and amplitude of a component.")
        if st.button("Calculate FFT", key="btn_manual"):
            fft_result = np.fft.fft(y_combined)
            fft_freq = np.fft.fftfreq(len(t), 1/fs)
            magnitudes = np.abs(fft_result) / (fs / 2)
            pos_mask = fft_freq > 0
            
            fig_freq1, ax_freq1 = plt.subplots(figsize=(10, 2.5))
            ax_freq1.stem(fft_freq[pos_mask], magnitudes[pos_mask], basefmt="b-")
            ax_freq1.set_xlim(0, max(60, (num_waves * 10) + 10)) # Auto-scale the X axis
            ax_freq1.set_xlabel("Frequency (Hz)")
            ax_freq1.set_ylabel("Magnitude")
            ax_freq1.grid(True, alpha=0.3)
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
        ax_time2.plot(t_audio, audio_data, color='#28B463', linewidth=0.5)
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
            ax_freq2.plot(fft_freq_audio, magnitudes_audio, color='#E74C3C') 
            ax_freq2.set_xlim(0, 4000) # Most human voice data is below 4000 Hz
            ax_freq2.set_xlabel("Frequency (Hz)")
            ax_freq2.set_ylabel("Magnitude")
            ax_freq2.grid(True, alpha=0.3)
            st.pyplot(fig_freq2)