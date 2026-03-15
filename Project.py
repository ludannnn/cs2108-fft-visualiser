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
    st.header("Build a Signal")
    st.write("Add multiple sine and cosine waves together to see how they form a waveform in the time domain.")
    
    col_controls, col_plots = st.columns([1, 2])
    
    with col_controls:
        st.subheader("Wave Controls")
        num_waves = st.number_input("Number of components to add:", min_value=1, max_value=5, value=2)
        
        fs = 1000  # Sample rate
        t = np.linspace(0, 1, fs, endpoint=False)
        y_combined = np.zeros_like(t)
        
        #sliders based on user input
        for i in range(num_waves):
            st.markdown(f"**Component {i+1}**")
            wave_type = st.radio(f"Type", ["Sine", "Cosine"], key=f"type_{i}", horizontal=True)
            freq = st.slider(f"Frequency (Hz)", 1, 50, (i+1)*5, key=f"freq_{i}")
            amp = st.slider(f"Amplitude", 0.1, 10.0, 1.0, key=f"amp_{i}")
            
            # construct wave with formula -> y(t) = A * sin(2 * pi * f * t)
            if wave_type == "Sine":
                y_combined += amp * np.sin(2 * np.pi * freq * t)
            else:
                y_combined += amp * np.cos(2 * np.pi * freq * t)
            st.divider()

    with col_plots:
        # Plot Time Domain
        st.subheader("Time Domain")
        fig_time1, ax_time1 = plt.subplots(figsize=(10, 3))
        ax_time1.plot(t, y_combined, color='#2E86C1')
        ax_time1.set_xlabel("Time (seconds)")
        ax_time1.set_ylabel("Amplitude")
        ax_time1.grid(True, alpha=0.3)
        st.pyplot(fig_time1)
        

        # Plot Frequency Domain
        st.subheader("Frequency Domain")
        if st.button("Calculate FFT", key="btn_manual"):
            # Compute FFT
            fft_result = np.fft.fft(y_combined)
            fft_freq = np.fft.fftfreq(len(t), 1/fs)
            
            # Normalize and filter for positive frequencies
            magnitudes = np.abs(fft_result) / (fs / 2)
            pos_mask = fft_freq > 0
            
            fig_freq1, ax_freq1 = plt.subplots(figsize=(10, 3))
            ax_freq1.stem(fft_freq[pos_mask], magnitudes[pos_mask], basefmt="b-")
            ax_freq1.set_xlim(0, 60)
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