
import pandas as pd
import numpy as np
import librosa

from pathlib import Path

BASE = Path(r"C:\NeuroTraceAIFinal")

manifest = pd.read_csv(BASE / "audio_manifest_labeled.csv")

results = []

print("Total recordings:", len(manifest))

for index, row in manifest.iterrows():
    audio_path = Path(str(row["AudioPath"]))

    if not audio_path.exists():
        print("File not found:", audio_path)
        continue

    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        if len(audio) == 0:
            print("Empty audio:", audio_path)
            continue

        # Extract pitch
        pitches = librosa.yin(
            audio,
            fmin=60,
            fmax=500,
            sr=sr
        )

        # Extract loudness-related features
        rms = librosa.feature.rms(y=audio)[0]

        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr
        )[0]

        zero_crossing = librosa.feature.zero_crossing_rate(
            audio
        )[0]

        # Extract voice-quality features
        harmonic, percussive = librosa.effects.hpss(audio)

        harmonic_energy = np.mean(harmonic ** 2)
        noise_energy = np.mean(percussive ** 2)

        hnr_approx = 10 * np.log10(
            (harmonic_energy + 1e-10) /
            (noise_energy + 1e-10)
        )

        results.append({
            "ParticipantID": row["ParticipantID"],
            "Task": row["Task"],
            "Category": row["Category"],
            "Filename": row["Filename"],
            "Duration_sec": len(audio) / sr,
            "Pitch_mean": np.mean(pitches),
            "Pitch_std": np.std(pitches),
            "Pitch_median": np.median(pitches),
            "RMS_mean": np.mean(rms),
            "RMS_std": np.std(rms),
            "SpectralCentroid_mean": np.mean(spectral_centroid),
            "SpectralCentroid_std": np.std(spectral_centroid),
            "ZeroCrossing_mean": np.mean(zero_crossing),
            "HNR_approx": hnr_approx
        })

    except Exception as e:
        print("Error:", audio_path, "-", e)

    if (index + 1) % 100 == 0:
        print("Processed", index + 1, "of", len(manifest))

features_df = pd.DataFrame(results)

output_path = BASE / "audio_acoustic_features.csv"
features_df.to_csv(output_path, index=False)

print("\nExtraction complete!")
print("Successful recordings:", len(features_df))
print("Saved to:", output_path)
print(features_df.head())
