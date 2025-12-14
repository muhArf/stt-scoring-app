import streamlit as st
import tempfile
import os
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel
from pydub import AudioSegment

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
    dan mengembalikan path ke file .wav yang sudah diproses.
    """
    tmp_input_path = None
    processed_audio_path = None

    # Simpan file yang diunggah ke lokasi sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_input_path = tmp_file.name
        
    try:
        # 1. Konversi/Normalisasi audio menggunakan Pydub
        audio = AudioSegment.from_file(tmp_input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Simpan ke file WAV sementara untuk pemrosesan
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
            audio.export(tmp_wav_file.name, format="wav")
            processed_audio_path = tmp_wav_file.name

        # 2. Aplikasi Noise Reduction menggunakan soundfile
        data, rate = sf.read(processed_audio_path)
        
        # Pastikan data float sebelum noise reduction
        if data.dtype != 'float64':
             data = data.astype('float64')

        reduced_noise_audio = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
        sf.write(processed_audio_path, reduced_noise_audio, rate)
        
        return processed_audio_path

    except Exception as e:
        # Menampilkan tipe dan pesan error yang spesifik jika gagal
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