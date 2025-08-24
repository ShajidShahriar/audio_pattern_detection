import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import io



def load_and_preprocess_audio(file_buffer, low_pass, high_pass, sr = 22050):
    """Loads, resamples, filters, and normalizes an audio file from an in-memory buffer."""
    signal, _ = librosa.load(file_buffer, sr=sr, mono=True)
    
    nyquist = 0.5 * sr
    low = low_pass / nyquist
    high = high_pass / nyquist
    b, a = butter(5, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    normalized_signal = librosa.util.normalize(filtered_signal)
    return normalized_signal, sr

def perform_correlation(target_audio, template_audio):
    """Performs FFT cross-correlation and returns the normalized result."""
    correlation = signal.correlate(target_audio, template_audio, mode='valid', method='fft')
    correlation_normalized = correlation / np.max(np.abs(correlation))
    return correlation_normalized

def find_detections(correlation, threshold, distance):
    """Finds peaks in the correlation signal above a certain threshold."""
    peaks, _ = signal.find_peaks(correlation, height=threshold, distance=distance)
    return peaks


st.set_page_config(layout="wide")
st.title(" Audio Pattern Detection Tool (V1)")
st.markdown("Upload your audio files, tune the parameters, and find the repeated patterns!")

st.sidebar.header("Tuning Parameters")

st.sidebar.subheader("1. Upload Audio Files")
target_file = st.sidebar.file_uploader("Upload Target Song (.wav, .mp3)", type=['wav', 'mp3'])
template_file = st.sidebar.file_uploader("Upload Template Clip (.wav, .mp3)", type=['wav', 'mp3'])

st.sidebar.subheader("2. DSP Settings")
LOW_PASS_CUTOFF = st.sidebar.slider("Low Pass Cutoff (Hz)", 50, 500, 300)
HIGH_PASS_CUTOFF = st.sidebar.slider("High Pass Cutoff (Hz)", 1000, 5000, 3400)

st.sidebar.subheader("3. Detection Settings")
detection_threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.01)
if target_file and template_file:
    st.header("Results")
    
    st.subheader("Uploaded Audio")
    st.audio(target_file, format='audio/wav', start_time=0)
    st.audio(template_file, format='audio/wav', start_time=0)

    if st.button("Analyze Audio"):
        with st.spinner('Processing audio... This may take a moment.'):
            target_audio, sr = load_and_preprocess_audio(target_file, low_pass=LOW_PASS_CUTOFF, high_pass=HIGH_PASS_CUTOFF)
            template_audio, _ = load_and_preprocess_audio(template_file, low_pass=LOW_PASS_CUTOFF, high_pass=HIGH_PASS_CUTOFF)
            
            correlation = perform_correlation(target_audio, template_audio)
            
            min_dist = len(template_audio)
            peaks = find_detections(correlation, detection_threshold, min_dist)
            
            st.success(f"Analysis Complete! Found **{len(peaks)}** potential matches.")
            
            st.subheader("Detection Plot")
            fig, ax = plt.subplots(figsize=(15, 5))
            
            correlation_time_axis = np.linspace(0, len(target_audio) / sr, len(correlation))
            ax.plot(correlation_time_axis, correlation, color='orange', label='Similarity Score')
            ax.plot(correlation_time_axis[peaks], correlation[peaks], "rx", markersize=10, label=f'{len(peaks)} Detections')
            ax.hlines(detection_threshold, 0, correlation_time_axis[-1], color='blue', linestyle='--', label=f'Threshold ({detection_threshold})')
            ax.set_title('Detection Results')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Similarity Score')
            ax.legend()
            
            st.pyplot(fig)
            
            st.subheader("Timestamps of Detections")
            timestamps = [f"{p/sr:.2f}s" for p in peaks]
            st.text(', '.join(timestamps))

else:
    st.info("Please upload both a target song and a template clip to begin.")