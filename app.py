import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. SETUP TAMPILAN HALAMAN
st.set_page_config(page_title="Chef Maestro", page_icon="👨‍🍳", layout="centered")
st.title("👨‍🍳 Chef Maestro")
st.caption("Asisten Masak Cerdas berbasis AI & Resep Pilihan")

# 2. MENU SAMPING (SIDEBAR) UNTUK API KEY
with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_key = st.text_input("Masukkan Gemini API Key:", type="password")
    st.markdown("[Dapatkan API Key Gratis](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.write("Aplikasi ini membaca referensi dari Vector Database buku resep Anda.")

# 3. FUNGSI LOAD DATABASE (Di-cache agar tidak berat)
@st.cache_resource
def load_vectordb():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # Membaca folder database yang ikut di-upload
    return Chroma(persist_directory="./chef_vectordb", embedding_function=embedding_model)

# 4. LOGIKA CHAT
if api_key:
    # Inisialisasi API & Model
    genai.configure(api_key=api_key)
    # Kita gunakan 1.5-flash agar batas chat per menit lebih leluasa
    model = genai.GenerativeModel("gemini-1.5-flash") 
    
    # Load database & pencari
    vectordb = load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Siapkan memori riwayat obrolan
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya Chef Maestro. Ada resep yang ingin kamu pelajari hari ini? 🍳"}
        ]

    # Tampilkan chat sebelumnya
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Kotak Ketik Pertanyaan
    if prompt := st.chat_input("Tanya Chef Maestro di sini..."):
        # Tampilkan chat user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Proses jawaban Chef
        with st.chat_message("assistant"):
            with st.spinner("👨‍🍳 Sedang mengingat resep..."):
                try:
                    # Cari referensi di database
                    docs = retriever.invoke(prompt)
                    konteks = "\n\n---\n\n".join([d.page_content for d in docs])

                    # Persona & Prompt Lengkap
                    persona = """Kamu adalah Chef Maestro, chef profesional Indonesia.
                    Jawab pertanyaan berdasarkan konteks buku resep dengan hangat.
                    Format jawaban: 
                    🍳 [Penjelasan Utama]
                    💡 Tips Chef: [Saran Pro]
                    ⚠️ Hindari: [Kesalahan Umum]"""
                    
                    full_prompt = f"{persona}\n\nKONTEKS BUKU RESEP:\n{konteks}\n\nPERTANYAAN: {prompt}"

                    # Minta Gemini membalas
                    response = model.generate_content(full_prompt)
                    jawaban = response.text
                    
                    # Tampilkan & simpan balasan
                    st.write(jawaban)
                    st.session_state.messages.append({"role": "assistant", "content": jawaban})
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👈 Silakan masukkan API Key di menu sebelah kiri untuk memulai obrolan.")