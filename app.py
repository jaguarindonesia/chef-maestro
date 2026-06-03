# ==========================================
# SUNTIKAN SQLITE BARU (WAJIB DITARUH PALING ATAS!)
# ==========================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ==========================================
# IMPORT LIBRARY UTAMA
# ==========================================
import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. PENGATURAN TAMPILAN HALAMAN
# ==========================================
st.set_page_config(page_title="Chef Maestro", page_icon="👨‍🍳", layout="centered")
st.title("👨‍🍳 Chef Maestro")
st.caption("Asisten Masak Cerdas berbasis AI & Resep Pilihan")

# ==========================================
# 2. MENU SAMPING (SIDEBAR) UNTUK API KEY
# ==========================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_key = st.text_input("Masukkan Gemini API Key:", type="password")
    st.markdown("[Dapatkan API Key Gratis](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.write("Aplikasi ini membaca referensi dari Vector Database buku resep Anda.")

# ==========================================
# 3. FUNGSI LOAD DATABASE (Di-cache agar ringan)
# ==========================================
@st.cache_resource
def load_vectordb():
    # Menggunakan model embedding yang sama persis dengan di Colab
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # Membaca folder database lokal 'chef_vectordb' (Sudah diperbaiki namanya)
    return Chroma(persist_directory="./chef_vectordb", embedding_function=embedding_model)

# ==========================================
# 4. LOGIKA CHAT & PENCARIAN
# ==========================================
if api_key:
    # Inisialisasi API & Model Gemini 1.5 Flash
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash") 
    
    # Load database dan buat mesin pencari (retriever)
    vectordb = load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Siapkan memori riwayat obrolan jika belum ada
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya Chef Maestro. Ada resep yang ingin kamu pelajari hari ini? 🍳"}
        ]

    # Tampilkan riwayat chat sebelumnya di layar
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Kotak input untuk user mengetik pertanyaan
    if prompt := st.chat_input("Tanya Chef Maestro di sini..."):
        # 1. Tampilkan pertanyaan user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2. Proses jawaban AI
        with st.chat_message("assistant"):
            with st.spinner("👨‍🍳 Sedang mengingat resep..."):
                try:
                    # Cari potongan teks resep yang relevan dari database
                    docs = retriever.invoke(prompt)
                    konteks = "\n\n---\n\n".join([d.page_content for d in docs])

                    # Susun instruksi (Persona) + Konteks + Pertanyaan
                    persona = """Kamu adalah Chef Maestro, seorang chef profesional Indonesia.
                    Jawab pertanyaan berdasarkan konteks buku resep dengan hangat.
                    Format jawaban: 
                    🍳 [Penjelasan Utama]
                    💡 Tips Chef: [Saran Pro]
                    ⚠️ Hindari: [Kesalahan Umum]"""
                    
                    full_prompt = f"{persona}\n\nKONTEKS BUKU RESEP:\n{konteks}\n\nPERTANYAAN: {prompt}"

                    # Minta Gemini menghasilkan teks balasan
                    response = model.generate_content(full_prompt)
                    jawaban = response.text
                    
                    # Tampilkan jawaban di layar dan simpan ke memori
                    st.write(jawaban)
                    st.session_state.messages.append({"role": "assistant", "content": jawaban})
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
else:
    # Pesan jika API Key belum diisi
    st.info("👈 Silakan masukkan API Key di menu sebelah kiri untuk memulai obrolan.")