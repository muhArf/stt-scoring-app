import streamlit as st
import os
import tempfile
import time # Untuk simulasi progress bar
from stt_module import load_stt_model, process_audio, transcribe_audio
from scoring_module import load_embedding_model, compute_confidence_score, compute_rubric_score, get_rubric_data

def main():
    st.set_page_config(layout="centered", page_title="STT & Interview Scoring App")

    # --- Custom CSS (Sama seperti sebelumnya) ---
    st.markdown("""
        <style>
        .stApp {
            background-color: #f9f9ff;
            padding-top: 50px;
        }
        .header-title {
            color: #1a1a1a;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
        }
        .stButton>button {
            border-radius: 40px; 
            background-color: #000000;
            color: #ffffff;
            border: 1px solid #000000;
            padding: 10px 32px;
        }
        .stTextInput, .stTextArea, .stFileUploader, .stSelectbox {
            border-radius: 12px;
            padding: 10px;
            background-color: #ffffff;
        }
        .block-container {
            padding-top: 0rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="header-title">Aplikasi Penilaian Wawancara Otomatis</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Memuat semua model di awal
    stt_model = load_stt_model()
    embed_model = load_embedding_model()
    rubric_data_all = get_rubric_data()
    
    # --- Kolom Utama Aplikasi ---
    col1, col2 = st.columns([1, 1])
    transcribed_text = ""
    
    with col1:
        st.subheader("1. Speech-to-Text (STT)")
        uploaded_file = st.file_uploader("Unggah file audio (mp3, wav, flac, ogg)", type=['mp3', 'wav', 'flac', 'ogg'])

        if uploaded_file is not None:
            progress_bar = st.progress(0)
            
            # --- Proses STT ---
            st.info("Memproses audio...")
            
            # 1. Proses Audio (Normalisasi & Noise Reduction)
            progress_bar.progress(30)
            processed_audio_path = process_audio(uploaded_file)
            
            if processed_audio_path:
                st.audio(processed_audio_path, format='audio/wav')
                
                # 2. Transkripsi Audio
                progress_bar.progress(60)
                st.info("Melakukan transkripsi...")
                transcribed_text = transcribe_audio(stt_model, processed_audio_path)
                
                progress_bar.progress(100)
                st.success("Transkripsi Selesai!")
                st.text_area("Hasil Transkripsi", transcribed_text, height=150)
                
                # Membersihkan file audio yang sudah diproses
                if os.path.exists(processed_audio_path):
                    os.remove(processed_audio_path)

    with col2:
        st.subheader("2. Penilaian Jawaban")

        # Input Jawaban
        # Menggunakan session_state agar teks transkripsi tetap ada saat interaksi lain
        if uploaded_file is not None and transcribed_text:
             st.session_state.candidate_answer = transcribed_text
        
        candidate_answer = st.text_area(
            "Jawaban Kandidat", 
            value=st.session_state.get('candidate_answer', ''), 
            height=100, 
            help="Jika audio diunggah, teks ini akan terisi otomatis."
        )
        
        # Pilihan Pertanyaan/Rubrik
        question_key = st.selectbox(
            "Pilih Pertanyaan Wawancara",
            options=list(rubric_data_all.keys()),
            format_func=lambda x: f"{x}: {rubric_data_all[x]['description']}"
        )

        # Jawaban Referensi
        reference_answer = st.text_area("Jawaban Referensi (Kunci Jawaban)", 
                                        placeholder="Masukkan jawaban ideal untuk perbandingan Confidence Score.", height=100)
        
        if st.button("Hitung Skor"):
            if candidate_answer and reference_answer:
                with st.spinner("Menghitung Skor..."):
                    
                    # Hitung Skor Rubrik
                    rubric_score, reason = compute_rubric_score(candidate_answer, question_key)
                    
                    # Hitung Confidence Score
                    confidence_score = compute_confidence_score(embed_model, candidate_answer, reference_answer)
                    
                    st.markdown("---")
                    st.subheader("Hasil Penilaian")
                    
                    # Tampilkan Hasil
                    res_col1, res_col2 = st.columns(2)
                    
                    current_rubric = rubric_data_all[question_key]

                    with res_col1:
                        st.metric("Skor Rubrik", f"{rubric_score:.2f} / {current_rubric['rubric']}")
                        st.caption(f"**Alasan:** {reason}")
                        
                    with res_col2:
                        st.metric("Confidence Score", f"{confidence_score:.2f} %")
                        st.caption("Kesamaan Semantik dengan Jawaban Referensi.")
                    
                    # Peringatan Confidence
                    min_conf = current_rubric['min_confidence']
                    if confidence_score < min_conf:
                        st.warning(f"Confidence Score ({confidence_score:.2f}%) di bawah batas minimum yang disarankan ({min_conf}%).")
                    else:
                        st.info(f"Confidence Score ({confidence_score:.2f}%) di atas batas minimum yang disarankan ({min_conf}%).")
            else:
                st.warning("Mohon masukkan Jawaban Kandidat dan Jawaban Referensi untuk dinilai.")

if __name__ == '__main__':
    main()