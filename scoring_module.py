import streamlit as st
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

# Data Rubrik (Disimpan di module scoring, tapi bisa dipindahkan ke config.py jika kompleks)
RUBRIC_DATA = {
    "q1": {
        "rubric": 4,
        "keywords": ["data preprocessing", "model definition", "training", "evaluation"],
        "min_confidence": 75,
        "description": "Langkah-langkah membangun model"
    },
    "q2": {
        "rubric": 3,
        "keywords": ["transfer learning", "TensorFlow", "pengalaman", "proyek"],
        "min_confidence": 60,
        "description": "Pengalaman dengan Transfer Learning"
    },
    "q3": {
        "rubric": 2,
        "keywords": ["TensorFlow model", "istilah umum"],
        "min_confidence": 50,
        "description": "Deskripsi umum TensorFlow"
    },
}

@st.cache_resource
def load_embedding_model():
    """Memuat model Sentence Transformer hanya sekali."""
    with st.spinner('Memuat Model Semantic Similarity...'):
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    st.success("Model Similarity berhasil dimuat.")
    return embed_model

def get_rubric_data(key=None):
    """Mengembalikan semua data rubrik atau data rubrik spesifik."""
    if key:
        return RUBRIC_DATA.get(key)
    return RUBRIC_DATA

def normalize_text(text):
    """Normalisasi teks untuk scoring."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Hapus tanda baca
    return text

def compute_confidence_score(embed_model, text, reference_text):
    """Hitung Confidence Score (Semantic Similarity)."""
    if not text or not reference_text:
        return 0.0

    embeddings = embed_model.encode([normalize_text(text), normalize_text(reference_text)])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Skala dari [0, 1] ke [0, 100]
    confidence_score = similarity * 100
    return round(confidence_score, 2)

def compute_rubric_score(candidate_answer, rubric_key):
    """Hitung Rubric Score berdasarkan keyword matching (Fuzzy Matching)."""
    rubric_data = RUBRIC_DATA.get(rubric_key)
    if not rubric_data:
        return 0, "Kunci rubrik tidak ditemukan."

    keywords = rubric_data["keywords"]
    candidate_norm = normalize_text(candidate_answer)
    candidate_words = candidate_norm.split()
    
    hits = 0
    used_keywords = set()
    
    for kw in keywords:
        # Fuzzy matching dengan threshold 80%
        best_match = process.extractOne(kw, candidate_words, scorer=fuzz.ratio)
        if best_match and best_match[1] >= 80: 
            hits += 1
            used_keywords.add(kw)
    
    max_rubric = rubric_data["rubric"]
    score = (hits / len(keywords)) * max_rubric if keywords else 0
    score = min(max_rubric, score)

    reason = f"Keyword yang cocok ({hits}/{len(keywords)}): {', '.join(used_keywords)}. Rubrik Maksimal: {max_rubric}"

    return round(score, 2), reason