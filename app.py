import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("passing-grade.csv")

# Function to find similar programs based on name similarity
def get_similar_prodi(selected_prodi, data, n_similar=5):
    """
    Find similar program names using TF-IDF and cosine similarity
    """
    try:
        # Get all program names
        prodi_names = data["NAMA PRODI"].dropna().unique()
        
        # Create TF-IDF vectorizer for program names
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words=None,     # Keep all words for Indonesian text
            lowercase=True
        )
        
        # Fit and transform program names
        tfidf_matrix = vectorizer.fit_transform(prodi_names)
        
        # Find index of selected program
        selected_idx = np.where(prodi_names == selected_prodi)[0]
        if len(selected_idx) == 0:
            return pd.DataFrame()
        
        selected_idx = selected_idx[0]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(tfidf_matrix[selected_idx:selected_idx+1], tfidf_matrix).flatten()
        
        # Get indices of most similar programs (excluding the selected one)
        similar_indices = similarities.argsort()[::-1][1:n_similar+1]  # Skip first one (itself)
        
        # Get similar program names
        similar_names = prodi_names[similar_indices]
        
        # Filter dataframe to get similar programs
        similar_prodi = data[data["NAMA PRODI"].isin(similar_names)].copy()
        
        # Add similarity scores
        similarity_scores = similarities[similar_indices]
        similar_prodi_with_scores = []
        
        for name, score in zip(similar_names, similarity_scores):
            prodi_data = data[data["NAMA PRODI"] == name].copy()
            prodi_data["SIMILARITY_SCORE"] = round(score, 3)
            similar_prodi_with_scores.append(prodi_data)
        
        if similar_prodi_with_scores:
            result = pd.concat(similar_prodi_with_scores, ignore_index=True)
            return result[["PTN", "NAMA PRODI", "RATAAN", "SIMILARITY_SCORE"]].sort_values("SIMILARITY_SCORE", ascending=False)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error finding similar programs: {str(e)}")
        return pd.DataFrame()

# --- UI HEADER ---
st.set_page_config(page_title="Rekomendasi PTN & Prodi", layout="wide")

st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/190/190411.png" width="60">
        <h1 style="color:#FF6F61; margin-left: 15px;">Rekomendasi PTN & Prodi Berdasarkan Nilai Try Out</h1>
    </div>
    <p>Masukkan nilai rata-rata try out kamu, lalu filter berdasarkan PTN atau Prodi jika perlu.</p>
""", unsafe_allow_html=True)

# --- INPUT NILAI ---
nilai_input = st.number_input("Masukkan nilai rata-rata try out kamu:", min_value=0.0, max_value=1000.0, step=0.1)

# --- FILTER TAMBAHAN ---
st.markdown("### 🎯 Filter Opsional")
daftar_ptn = ["Semua PTN"] + sorted(df["PTN"].unique())
pilihan_ptn = st.selectbox("Pilih PTN (opsional):", daftar_ptn)

kata_kunci_prodi = st.text_input("Cari nama prodi (opsional):", "")

# --- FILTERING DATA ---
filtered_df = df.copy()

if pilihan_ptn != "Semua PTN":
    filtered_df = filtered_df[filtered_df["PTN"] == pilihan_ptn]

if kata_kunci_prodi:
    filtered_df = filtered_df[filtered_df["NAMA PRODI"].str.contains(kata_kunci_prodi, case=False, na=False)]

# Initialize recommendations variable
recommendations = pd.DataFrame()

# --- PROSES REKOMENDASI ---
if nilai_input > 0:
    col1, col2 = st.columns([1.2, 1.0], gap="large")

    # --- BAGIAN 1: TAMPILKAN SEMUA PRODI YANG DIBAWAH NILAI ---
    prodi_lolos = filtered_df[filtered_df["RATAAN"] <= nilai_input].sort_values(by="RATAAN", ascending=False)

    with col1:
        st.subheader("📋 Prodi dengan Passing Grade Di Bawah Nilai Kamu")
        st.dataframe(prodi_lolos[["PTN", "NAMA PRODI", "RATAAN"]].reset_index(drop=True), use_container_width=True)

    # --- BAGIAN 2: KNN REKOMENDASI DARI FILTER YANG DIPILIH ---
    if not filtered_df.empty:
        model = NearestNeighbors(n_neighbors=min(10, len(filtered_df)), metric="euclidean")
        model.fit(filtered_df[["RATAAN"]])
        distances, indices = model.kneighbors(np.array([[nilai_input]]))
        recommendations = filtered_df.iloc[indices[0]]
    else:
        recommendations = pd.DataFrame()

    with col2:
        st.subheader("🤖 Rekomendasi Program Studi Terdekat")
        if not recommendations.empty:
            st.dataframe(recommendations[["PTN", "NAMA PRODI", "RATAAN"]].reset_index(drop=True), use_container_width=True)
        else:
            st.warning("Tidak ada prodi yang sesuai dengan filter yang dipilih.")

    st.success(f"Menampilkan rekomendasi berdasarkan nilai {nilai_input}")

    # --- BAGIAN 3: TEMUKAN PRODI MIRIP ---
    if not recommendations.empty:
        st.markdown("### 🔁 Temukan Prodi Mirip")
        
        # Get unique program names from recommendations
        available_prodi = recommendations["NAMA PRODI"].unique()
        
        if len(available_prodi) > 0:
            selected_prodi = st.selectbox("Pilih salah satu prodi dari rekomendasi:", available_prodi)
            
            if selected_prodi:
                with st.spinner("Mencari prodi yang mirip..."):
                    similar_prodi = get_similar_prodi(selected_prodi, df)
                    
                if not similar_prodi.empty:
                    st.write(f"Prodi mirip dengan **{selected_prodi}**:")
                    st.dataframe(similar_prodi.reset_index(drop=True), use_container_width=True)
                    
                    # Show explanation
                    st.info("💡 **Catatan:** Similarity Score menunjukkan seberapa mirip nama prodi (1.0 = sangat mirip, 0.0 = tidak mirip)")
                else:
                    st.warning("Tidak ditemukan prodi mirip.")
        else:
            st.info("Tidak ada rekomendasi prodi untuk mencari yang mirip.")

else:
    st.info("Silakan masukkan nilai try out kamu terlebih dahulu.")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            font-family: 'Segoe UI', sans-serif;
            color: #333;
            padding: 1rem;
        }
        h1, h2, h3, .stMarkdown h1 {
            color: #FF6F61 !important;
        }
        .stDataFrame {
            border-radius: 10px;
        }
        .stNumberInput input {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)