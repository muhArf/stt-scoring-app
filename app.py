import streamlit as st
import os
import tempfile
import time
from stt_module import load_stt_model, process_audio, transcribe_audio
from scoring_module import load_embedding_model, compute_confidence_score, compute_rubric_score, get_rubric_data
from config import RUBRIC_DATA # Import data rubrik dari config

def main():
    st.set_page_config(layout="centered", page_title="Aplikasi Penilaian Wawancara ML")

    # --- Custom CSS untuk Estetika ---
    st.markdown("""
        <style>
        .stApp { background-color: #f9f9ff; padding-top: 20px;}
        .header-title { color: #1a1a1a; font-size: 36px; font-weight: 700; text-align: center;}
        .stButton>button { border-radius: 40px; background-color: #000000; color: #ffffff; padding: 10px 32px; transition: background-color 0.3s;}
        .stButton>button:hover { background-color: #333333;}
        .stTextArea, .stFileUploader, .stSelectbox { border-radius: 12px; padding: 10px; background-color: #ffffff;}
        .main-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="header-title">Sistem Penilaian Wawancara AI</div>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Load Models (Cepat karena di-cache) ---
    stt_model = load_stt_model()
    embed_model = load_embedding_model()
    
    # --- Sidebar: Pemilihan Pertanyaan ---
    with st.sidebar:
        st.header("Konfigurasi Wawancara")
        
        # Pilihan Pertanyaan/Rubrik
        question_options = {k: v['title'] for k, v in RUBRIC_DATA.items()}
        question_key = st.selectbox(
            "Pilih Pertanyaan Wawancara",
            options=list(question_options.keys()),
            format_func=lambda x: question_options[x]
        )
        
        current_rubric = RUBRIC_DATA[question_key]
        
        st.subheader("Detail Rubrik Terpilih")
        st.info(f"Poin Maksimum: **{current_rubric['rubric']}**")
        st.info(f"Keyword Kunci: {', '.join(current_rubric['keywords'])}")
        st.caption(f"Confidence Minimum Disarankan: {current_rubric['min_confidence']}%")

    # --- Main Content: Upload dan Proses ---
    st.subheader(f"Pertanyaan: {current_rubric['title']}")
    st.write("Silakan unggah rekaman jawaban kandidat (Audio/Video).")
    
    # Kolom untuk Upload dan Hasil
    upload_col, result_col = st.columns(2)
    transcribed_text = ""
    
    with upload_col:
        # File Uploader
        uploaded_file = st.file_uploader(
            "Unggah File Jawaban (.mp3, .wav, .mp4, dll.)", 
            type=['mp3', 'wav', 'flac', 'ogg', 'mp4', 'mov'] # Tambah dukungan video
        )
        
        if uploaded_file is not None:
            
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.caption("Audio/Video Input:")
            # Tampilkan kontrol media
            st.audio(uploaded_file, format=uploaded_file.type)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("PROSES STT & HITUNG SKOR", use_container_width=True):
                
                # --- PROSES STT (Langkah 1) ---
                with st.spinner("1/2: Memproses audio dan Noise Reduction..."):
                    # Panggil fungsi processing dari stt_module
                    processed_audio_path = process_audio(uploaded_file)
                
                if processed_audio_path:
                    # --- TRANSKRIPSI (Langkah 2) ---
                    with st.spinner("2/2: Melakukan Transkripsi dan Menghitung Skor..."):
                        transcribed_text = transcribe_audio(stt_model, processed_audio_path)
                        
                        # Simpan hasil untuk digunakan di kolom skor
                        st.session_state.transcribed_text = transcribed_text
                        st.session_state.current_question = question_key

                    # Hapus file sementara setelah selesai
                    if os.path.exists(processed_audio_path):
                        os.remove(processed_audio_path)

                    st.toast("Transkripsi Selesai!")
                    st.rerun() # Refresh untuk menampilkan hasil di kolom kedua
        
    with result_col:
        # --- TAMPILAN HASIL ---
        if st.session_state.get('transcribed_text') and st.session_state.get('current_question') == question_key:
            
            transcribed_text = st.session_state.transcribed_text
            
            # --- 1. Hasil Transkripsi ---
            st.subheader("Hasil Transkripsi")
            st.text_area("Jawaban Kandidat (Dapat Diedit)", transcribed_text, height=150, key="candidate_answer_input")
            
            # --- 2. Perhitungan Skor ---
            if st.button("TAMPILKAN SKOR AKHIR", key="score_button", use_container_width=True):
                
                final_answer = st.session_state.candidate_answer_input
                
                with st.spinner("Menghitung Similarity dan Rubrik..."):
                    
                    # Ambil jawaban referensi dari config
                    reference_answer = current_rubric['reference_answer']
                    
                    # Panggil fungsi scoring
                    rubric_score, reason = compute_rubric_score(final_answer, question_key)
                    confidence_score = compute_confidence_score(embed_model, final_answer, reference_answer)
                    
                    st.markdown("---")
                    st.subheader("✅ Penilaian Otomatis")
                    
                    # Kotak Hasil
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.metric("Skor Rubrik", f"{rubric_score:.2f} / {current_rubric['rubric']}", 
                                  help=f"Dihitung berdasarkan kecocokan keyword: {', '.join(current_rubric['keywords'])}")
                        
                    with res_col2:
                        st.metric("Confidence Score", f"{confidence_score:.2f} %", 
                                  help="Kesamaan semantik antara jawaban kandidat dan jawaban referensi.")

                    st.markdown(f"**Ringkasan Rubrik:** {reason}")

                    # Peringatan Confidence
                    min_conf = current_rubric['min_confidence']
                    if confidence_score < min_conf:
                        st.warning(f"Confidence Score ({confidence_score:.2f}%) di bawah batas minimum yang disarankan ({min_conf}%). Perlu tinjauan manual.")
                    else:
                        st.success(f"Confidence Score ({confidence_score:.2f}%) OK. Kesamaan tinggi dengan jawaban ideal.")
                        
                # Tampilkan Jawaban Referensi
                with st.expander("Lihat Jawaban Referensi"):
                    st.caption("Jawaban Ideal (untuk perbandingan Confidence):")
                    st.write(reference_answer)


# Inisialisasi session state jika belum ada
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

if __name__ == '__main__':
    main()
