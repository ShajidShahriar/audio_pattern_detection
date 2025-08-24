import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# --- SETTINGS ---
TARGET_SR = 22050
TARGET_FILE = 'samples/inhale_exhale_acapella.wav'
TEMPLATE_FILE = 'samples/too_high_tempV2.wav'

# --- GHOST AND TARGET TIMES (in seconds) ---
# We'll grab the audio from these specific moments
GHOST_TIME_SEC = 64.0  # 1 minute 4 seconds
MISSED_TIME_SEC = 38.0

# --- LOAD FILES ---
print("Loading audio files...")
target_audio, sr = librosa.load(TARGET_FILE, sr=TARGET_SR, mono=True)
template_audio, _ = librosa.load(TEMPLATE_FILE, sr=TARGET_SR, mono=True)

# --- FUNCTION TO EXTRACT CLIPS ---
def get_clip(audio, sr, start_sec, duration_sec):
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    return audio[start_sample:end_sample]

print("Extracting audio clips for comparison...")
template_duration = len(template_audio) / sr
ghost_clip = get_clip(target_audio, sr, GHOST_TIME_SEC, template_duration)
missed_clip = get_clip(target_audio, sr, MISSED_TIME_SEC, template_duration)

# --- FUNCTION TO COMPUTE SPECTROGRAMS ---
def compute_spectrogram(audio, sr):
    stft = librosa.stft(audio)
    # Convert amplitude to decibels (a more human-like scale)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return spectrogram

print("Computing spectrograms...")
template_spec = compute_spectrogram(template_audio, sr)
ghost_spec = compute_spectrogram(ghost_clip, sr)
missed_spec = compute_spectrogram(missed_clip, sr)

# --- PLOT FOR COMPARISON ---
print("Generating comparison plot...")
fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

librosa.display.specshow(template_spec, sr=sr, x_axis='time', y_axis='log', ax=ax[0])
ax[0].set_title(f'1. Your Template (from file)')

librosa.display.specshow(missed_spec, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
ax[1].set_title(f'2. Missed Match (audio at {MISSED_TIME_SEC}s)')

librosa.display.specshow(ghost_spec, sr=sr, x_axis='time', y_axis='log', ax=ax[2])
ax[2].set_title(f'3. Ghost Peak (audio at {GHOST_TIME_SEC}s)')

fig.suptitle('Spectrogram Comparison: The Fingerprint of the Sound', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()