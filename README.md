# ⚡ NVIDIA NIM RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application built with **LangChain**, **NVIDIA NIM**, and **Streamlit** that lets you ask questions against your own PDF documents using FAISS vector search.

---

## 🚀 Features

- 📄 Load and parse PDF documents from a local directory
- 🧠 Generate embeddings using NVIDIA NIM Embeddings
- ⚡ Fast similarity search powered by FAISS
- 🤖 LLM responses via `meta/llama-3.1-70b-instruct` through NVIDIA NIM
- 🖥️ Clean Streamlit UI with document context expander

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io/) | Frontend UI |
| [LangChain](https://www.langchain.com/) | RAG orchestration |
| [NVIDIA NIM](https://developer.nvidia.com/nim) | LLM + Embeddings |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector store |
| [PyPDF](https://pypdf.readthedocs.io/) | PDF loading |

---
   

## ▶️ Usage

```bash
streamlit run app.py
```

1. Click **"Document Embedding"** to load and vectorize your PDFs.
2. Type your question in the input box.
3. View the answer and the relevant document chunks used to generate it.

---

## 📁 Project Structure

```
nvidia-nim-rag-qa/
│
├── app.py                  # Main Streamlit application
├── us_census/              # Directory containing PDF documents
├── .env                    # Environment variables (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📋 Requirements

```
streamlit
langchain
langchain-nvidia-ai-endpoints
langchain-community
langchain-text-splitters
faiss-cpu
pypdf
python-dotenv
```
