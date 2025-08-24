import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt

# SETTINGS 
LOW_PASS_CUTOFF = 300
HIGH_PASS_CUTOFF = 3400
TARGET_SR = 22050

def load_and_preprocess_audio(file_path):
    """Loads, resamples, filters, and normalizes an audio file."""
    signal, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    nyquist = 0.5 * sr
    low = LOW_PASS_CUTOFF / nyquist
    high = HIGH_PASS_CUTOFF / nyquist
    b, a = butter(5, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    normalized_signal = librosa.util.normalize(filtered_signal)
    return normalized_signal, sr

target_file = 'samples/inhale_exhale.wav' # Use the acapella or the full song
template_file = 'samples/inhale_exhale_temp.wav' # Your template

target_audio, sr = load_and_preprocess_audio(target_file)
template_audio, _ = load_and_preprocess_audio(template_file)
print("Loaded and processed audio files.")

# --- CROSS-CORRELATION ---
print("\nPerforming FFT cross-correlation...")
correlation = signal.correlate(target_audio, template_audio, mode='valid', method='fft') #janina kmne
correlation_normalized = correlation / np.max(np.abs(correlation))
print("Correlation complete.")


print("\n--- Running Peak Detection ---")
detection_threshold = 0.18
min_distance_between_peaks = len(template_audio)

peaks, _ = signal.find_peaks(correlation_normalized,
                              height=detection_threshold,
                              distance=min_distance_between_peaks)

print(f"\n--- DETECTION RESULTS ---")
print(f"Found {len(peaks)} instances of the phrase above a threshold of {detection_threshold}.")

print("\nGenerating final plot with detections...")

fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# Plot 1: The Target Audio
librosa.display.waveshow(target_audio, sr=sr, ax=ax[0], color='green')
ax[0].set_title('Filtered Target Audio')
ax[0].set_ylabel('Amplitude')

# Plot 2: The Correlation Result with Detections
correlation_time_axis = np.linspace(0, len(target_audio) / sr, len(correlation_normalized))
ax[1].plot(correlation_time_axis, correlation_normalized, color='orange', label='Similarity Score')
ax[1].plot(correlation_time_axis[peaks], correlation_normalized[peaks], "rx", markersize=10, label=f'{len(peaks)} Detections')
ax[1].hlines(detection_threshold, 0, correlation_time_axis[-1], color='blue', linestyle='--', label=f'Threshold ({detection_threshold})')
ax[1].set_title('Final Detections')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Similarity Score')
ax[1].legend()

plt.tight_layout()
plt.show()