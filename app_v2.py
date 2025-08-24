import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, find_peaks
 

def load_audio(file_path, sr=22050):
    """Loads an audio file from a file path."""
    signal, _ = librosa.load(file_path, sr=sr, mono=True)
    return signal, sr

def extract_mfccs(audio, sr, n_mfcc):
    """Computes MFCCs from an audio signal."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
    return mfccs

def dtw_search(target_mfccs, template_mfccs):
    """Performs a sliding window DTW search."""
    n_template_frames = template_mfccs.shape[1]
    n_target_frames = target_mfccs.shape[1]
    dtw_costs = []
    progress_bar = st.progress(0, text="Performing DTW search...")

    for i in range(n_target_frames - n_template_frames):
        window = target_mfccs[:, i : i + n_template_frames]
        D, _ = librosa.sequence.dtw(X=template_mfccs, Y=window, metric='euclidean')
        dtw_costs.append(D[-1, -1])
        progress_bar.progress((i + 1) / (n_target_frames - n_template_frames), text="Performing DTW search...")
        
    progress_bar.empty()
    return np.array(dtw_costs)

def find_detections(dtw_costs, threshold, min_dist_frames):
    """Finds detections by looking for valleys in the DTW cost."""
    min_cost = np.min(dtw_costs)
    max_cost = np.max(dtw_costs)
    normalized_costs = (dtw_costs - min_cost) / (max_cost - min_cost)
    inverted_costs = 1 - normalized_costs
    
    peaks, _ = find_peaks(inverted_costs, height=threshold, distance=min_dist_frames)
    return peaks, inverted_costs


st.set_page_config(layout="wide")
st.title("Audio Pattern Detection Tool (V2 - Advanced)")
st.markdown("Using MFCCs and Dynamic Time Warping for robust detection.")

st.sidebar.header("V2 Tuning Parameters")
target_file = st.sidebar.file_uploader("Upload Target Song (use .mp3 for large files)", type=['wav', 'mp3'])
template_file = st.sidebar.file_uploader("Upload Template Clip", type=['wav', 'mp3'])
st.sidebar.subheader("Feature Settings")
n_mfcc = st.sidebar.slider("Number of MFCCs (Detail level)", 5, 40, 13)
st.sidebar.subheader("Detection Settings")
detection_threshold = st.sidebar.slider("Detection Threshold (Higher is stricter)", 0.0, 1.0, 0.7, 0.01)

if target_file and template_file:
    temp_dir = "temp_audio"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    target_path = os.path.join(temp_dir, target_file.name)
    template_path = os.path.join(temp_dir, template_file.name)

    with open(target_path, "wb") as f:
        f.write(target_file.getbuffer())
    with open(template_path, "wb") as f:
        f.write(template_file.getbuffer())

    st.header("Uploaded Audio")
    st.audio(target_path)
    st.audio(template_path)

    if st.button("Analyze Audio (Advanced)"):
        with st.spinner('Performing advanced analysis... This WILL be slow on long files.'):
            target_audio, sr = load_audio(target_path)
            template_audio, _ = load_audio(template_path)
            
            target_mfccs = extract_mfccs(target_audio, sr, n_mfcc)
            template_mfccs = extract_mfccs(template_audio, _, n_mfcc)
            
            dtw_costs = dtw_search(target_mfccs, template_mfccs)
            
            min_dist_frames = template_mfccs.shape[1]
            peaks, inverted_costs = find_detections(dtw_costs, detection_threshold, min_dist_frames)
            
            st.success(f"Analysis Complete!")
            st.metric(label="Potential Matches Found", value=len(peaks))
            
            st.subheader("DTW Cost Plot")
            fig, ax = plt.subplots(figsize=(15, 5))
            
            hop_length = 512 # This is the default for librosa's MFCC
            time_axis = librosa.frames_to_time(np.arange(len(inverted_costs)), sr=sr, hop_length=hop_length)
            
            ax.plot(time_axis, inverted_costs, color='purple', label='Match Score (1 is best)')
            ax.plot(time_axis[peaks], inverted_costs[peaks], "rx", markersize=10, label=f'{len(peaks)} Detections')
            ax.hlines(detection_threshold, 0, time_axis[-1], color='blue', linestyle='--', label=f'Threshold ({detection_threshold})')
            ax.set_title('Detection Results using DTW')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized Score')
            ax.legend()
            st.pyplot(fig)
            
            # --- NEW: Display Timestamps ---
            st.subheader("Timestamps of Detections")
            peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
            timestamps_str = [f"{t:.2f}s" for t in peak_times]
            st.text(', '.join(timestamps_str))
            
else:
    st.info("Please upload both a target song and a template clip to begin.")