# PDF_CHATBOT
A Streamlit-based PDF chatbot that lets users upload documents and ask questions using LangChain, Hugging Face embeddings, and FAISS.
# My PDF Chatbot

I built this because I wanted to understand how to make documents "talk" to me. It's a simple, honest project that helps you upload a PDF, chat with it, and find the specific information you need without scrolling through pages.

## Why I built this
I was tired of searching through long PDFs for one specific detail. I wanted to see if I could use RAG (Retrieval-Augmented Generation) to make document searching feel more like a conversation.

## What it does
- **Upload your own PDFs**: Just drop in your file and start asking.
- **Local Embeddings**: It uses Hugging Face to process your text locally.
- **Fast Search**: Powered by FAISS so it gets you answers quickly.

## How to get it running
1. Clone this repo: `git clone <your-repo-link>`
2. Create a virtual environment and activate it.
3. Install what you need: `pip install -r requirements.txt`
4. Run it: `python -m streamlit run app.py`

## Learning moments
I learned a ton about how vector databases work and why splitting text properly is actually harder than it sounds! If you have any suggestions on how to make the retrieval better, feel free to open an issue.
