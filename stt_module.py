import streamlit as st
import tempfile
import os
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel
import librosa  # Menggantikan pydub
import numpy as np

# Configuration
MODEL_SIZE = "small" 
DEVICE = "cpu" 
COMPUTE_TYPE = "int8"

@st.cache_resource
def load_stt_model():
    """Memuat model STT Whisper hanya sekali."""
    with st.spinner(f'Memuat Model Whisper ({MODEL_SIZE})...'):
        stt_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    st.success("Model Speech-to-Text berhasil dimuat.")
    return stt_model

def process_audio(uploaded_file):
    """
    Menyimpan file yang diunggah, melakukan pra-pemrosesan (Normalisasi & Noise Reduction), 
    dan mengembalikan path ke file .wav yang sudah diproses menggunakan librosa.
    """
    tmp_input_path = None
    processed_audio_path = None

    # 1. Simpan file yang diunggah ke lokasi sementara untuk dibaca librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_input_path = tmp_file.name
        
    try:
        # 2. Muat audio menggunakan Librosa
        # target_sr=16000: Resampling ke 16kHz (standar STT)
        # mono=True: Konversi ke mono
        st.info("Memproses audio (Normalisasi dan Resampling)...")
        audio_data, sr = librosa.load(tmp_input_path, sr=16000, mono=True)
        
        # 3. Aplikasi Noise Reduction
        st.info("Melakukan Noise Reduction...")
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sr, prop_decrease=0.8)
        
        # 4. Simpan hasil ke file WAV sementara untuk Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
            sf.write(tmp_wav_file.name, reduced_noise_audio, sr)
            processed_audio_path = tmp_wav_file.name
        
        return processed_audio_path

    except Exception as e:
        st.error(f"Error saat memproses audio: {type(e).__name__}: {e}")
        return None
    finally:
        # Selalu bersihkan file input sementara
        if tmp_input_path and os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)


def transcribe_audio(stt_model, audio_path):
    """Melakukan transkripsi audio menggunakan model Whisper."""
    try:
        segments, info = stt_model.transcribe(
            audio_path,
            language="id", 
            task="transcribe",
            temperature=0,
            beam_size=5,
        )
        
        transcribed_text = " ".join(segment.text for segment in segments)
        return transcribed_text

    except Exception as e:
        st.error(f"Error saat transkripsi: {e}")
        return None
