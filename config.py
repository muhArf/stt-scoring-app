# config.py

# Daftar lengkap rubrik pertanyaan wawancara
RUBRIC_DATA = {
    "q1_basic_steps": {
        "title": "Q1: Langkah-Langkah Membangun Model ML",
        "rubric": 4,
        "keywords": ["data preprocessing", "model definition", "training", "evaluation", "deployment"],
        "min_confidence": 75,
        "reference_answer": "Prosesnya dimulai dari data preprocessing seperti pembersihan dan normalisasi. Kemudian, definisi arsitektur model (misalnya, Keras Sequential), kompilasi model, melatih model dengan data training, dan terakhir mengevaluasi kinerja model menggunakan metrik yang tepat."
    },
    "q2_transfer_learning": {
        "title": "Q2: Pengalaman dengan Transfer Learning",
        "rubric": 3,
        "keywords": ["transfer learning", "fine-tuning", "pre-trained", "VGG16", "EfficientNet"],
        "min_confidence": 60,
        "reference_answer": "Saya pernah menggunakan Transfer Learning, khususnya untuk tugas Computer Vision, dengan mengambil model pre-trained seperti VGG16 atau EfficientNet. Saya melakukan fine-tuning pada layer akhir untuk menyesuaikan model dengan dataset proyek saya yang spesifik."
    },
    "q3_tensorflow_concept": {
        "title": "Q3: Konsep Dasar TensorFlow",
        "rubric": 2,
        "keywords": ["TensorFlow", "library", "komputasi numerik", "graph"],
        "min_confidence": 50,
        "reference_answer": "TensorFlow adalah library open-source yang digunakan untuk komputasi numerik berkinerja tinggi. Fokus utamanya adalah pada pembuatan dan pelatihan model Machine Learning, menggunakan struktur data berbasis graph."
    },
    "q4_dropout_implementation": {
        "title": "Q4: Implementasi Dropout",
        "rubric": 4,
        "keywords": ["dropout", "overfitting", "tf.keras.layers.Dropout", "deactivate neurons"],
        "min_confidence": 80,
        "reference_answer": "Dropout diimplementasikan menggunakan tf.keras.layers.Dropout dengan rate tertentu. Tujuannya adalah mencegah overfitting dengan secara acak menonaktifkan sejumlah neuron pada setiap iterasi training, memaksa jaringan untuk tidak terlalu bergantung pada neuron tertentu."
    },
    # Anda bisa menambahkan pertanyaan lain di sini:
    "q5_cnn_process": {
        "title": "Q5: Proses Pembangunan CNN",
        "rubric": 3,
        "keywords": ["Convolutional Layer", "Pooling Layer", "Flatten", "Dense", "feature extraction"],
        "min_confidence": 70,
        "reference_answer": "Membangun CNN melibatkan lapisan konvolusi untuk ekstraksi fitur, diikuti oleh lapisan pooling untuk mengurangi dimensi. Kemudian, outputnya di-flatten sebelum masuk ke lapisan dense untuk klasifikasi akhir."
    }
}
