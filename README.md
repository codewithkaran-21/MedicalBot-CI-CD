# 🏥 MedicalBot Agent (LangGraph + Gemini + Pinecone)

A Retrieval-Augmented Generation (RAG) based **Medical Chatbot** built with **LangGraph**, **LangChain**, and **Google Gemini**.  
It retrieves contextually relevant medical data and generates verified answers, followed by a reflection/validation step.

---

## 🚀 Features
- Multi-node agent graph (Plan → Retrieve → Answer → Reflect)
- Medical context retrieval using **Pinecone** + **Sentence Transformers**
- Answer generation via **Google Gemini**
- Reflection validation for accuracy and relevance
- Flask-based web UI
- Supports automated evaluation via BLEU, ROUGE, or LLM-as-a-Judge

---

## 🧩 Setup Instructions

### **1️⃣ Clone the Repo**
```bash
git clone https://github.com/yourusername/MedicalBot-CI-CD.git
cd MedicalBot-CI-CD


conda create -n medibot python=3.10
conda activate medibot
pip install -r requirements.txt


3️⃣ Environment Variables

Create a .env file:
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_region


4️⃣ Run the App
python src/app.py
