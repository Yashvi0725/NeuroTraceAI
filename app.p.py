
import io

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NeuroTrace AI",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ NeuroTrace AI")
st.subheader("Voice Biomarker Research Demo")

st.write(
    "Analyze basic acoustic properties of a voice recording "
    "for neurological research exploration."
)

st.warning(
    "Research prototype only. This application does not diagnose "
    "ALS or any other neurological condition."
)

with st.sidebar:
    st.header("Project Information")
    st.write("Project: NeuroTrace AI")
    st.write("Model: Minsk2020 Random Forest")
    st.write("Status: Audio analysis demo")
    st.info(
        "The trained model is not connected to predictions yet. "
        "We must first reproduce its original 131 acoustic features."
    )

st.markdown("## 1. Provide a voice sample")

col1, col2 = st.columns(2)

with col1:
    recording = st.audio_input("Record your voice")

with col2:
    uploaded = st.file_uploader(
        "Or upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"]
    )

audio = recording if recording is not None else uploaded

if audio is None:
    st.info("Record or upload a voice sample to begin.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", "--")
    col2.metric("Pitch", "--")
    col3.metric("RMS Energy", "--")

    st.markdown("### Analysis features")
    st.write(
        "- Voice recording and playback\n"
        "- Audio duration\n"
        "- Estimated pitch\n"
        "- Signal energy and zero-crossing rate\n"
        "- Waveform visualization"
    )

else:
    try:
        audio_bytes = audio.getvalue()

        y, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )

        if len(y) == 0:
            st.error("The audio file is empty.")
            st.stop()

        duration = len(y) / sr

        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr
        )[0]

        pitch = np.array([])

        if len(y) >= 2048:
            pitch = librosa.yin(
                y,
                fmin=65,
                fmax=400,
                sr=sr,
                frame_length=2048
            )

            pitch = pitch[
                np.isfinite(pitch)
                & (pitch >= 65)
                & (pitch <= 400)
            ]

        median_pitch = (
            float(np.median(pitch))
            if len(pitch) > 0
            else np.nan
        )

        st.markdown("## 2. Voice recording")

        st.audio(audio_bytes)

        if duration < 2:
            st.warning(
                "This recording is short. Try a 5–10 second sample "
                "for a more useful demonstration."
            )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Duration", f"{duration:.2f} seconds")

        c2.metric(
            "Estimated pitch",
            f"{median_pitch:.1f} Hz"
            if np.isfinite(median_pitch)
            else "Unavailable"
        )

        c3.metric("Mean RMS energy", f"{np.mean(rms):.4f}")

        c4.metric("Zero-crossing rate", f"{np.mean(zcr):.4f}")

        st.markdown("## 3. Acoustic measurements")

        measurements = pd.DataFrame({
            "Measurement": [
                "Sample rate",
                "Mean spectral centroid",
                "Mean RMS energy",
                "Mean zero-crossing rate"
            ],
            "Value": [
                f"{sr} Hz",
                f"{np.mean(centroid):.2f} Hz",
                f"{np.mean(rms):.4f}",
                f"{np.mean(zcr):.4f}"
            ]
        })

        st.dataframe(
            measurements,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("## 4. Voice waveform")

        time = np.arange(len(y)) / sr

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(time, y)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Recorded Voice Signal")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("## 5. AI model integration")

        st.info(
            "Audio analysis completed. The trained Minsk2020 model "
            "is not yet producing predictions because these general "
            "measurements do not match its required 131 dataset features."
        )

        st.caption(
            "Pitch and other measurements depend on the speaker, "
            "microphone, recording environment, and analysis method. "
            "They are not diagnostic results."
        )

    except Exception as error:
        st.error("Could not analyze this audio file.")
        st.write(
            "Try recording directly in the app or uploading a WAV file."
        )
        st.code(str(error))
