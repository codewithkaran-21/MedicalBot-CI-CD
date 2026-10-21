# MedicalBot Agent

A Retrieval-Augmented Generation (RAG) based medical chatbot that answers health-related questions using a combination of vector search, the Google Gemini model, and a LangGraph-based multi-step reasoning workflow.

The project is designed to retrieve relevant medical context from a knowledge base built from PDF documents, then generate a concise answer and perform a lightweight reflection step to validate whether the answer directly addresses the user query.

This repository is a working prototype for an AI medical assistant with a Flask web interface and Pinecone-backed semantic retrieval.

---

## Project overview

MedicalBot Agent is a Python application that:

- loads medical PDF documents from the `data/` folder,
- splits the text into manageable chunks,
- converts chunks into embeddings using a Hugging Face sentence-transformer model,
- stores those vectors in Pinecone,
- retrieves the most relevant documents based on a user question,
- passes the retrieved context and the question into a Gemini LLM,
- runs a reflection step to decide whether the answer is relevant and directly addresses the question,
- presents the response in a simple chatbot UI built with Flask.

The core of the system is implemented using LangChain and LangGraph patterns, while the user-facing surface is a basic web app.

---

## Repository structure

```text
MedicalBot-CI-CD/
├── .gitignore
├── LICENSE
├── Procfile
├── README.md
├── project_report.txt
├── requirements.txt
├── setup.py
├── store_index.py
├── data/
│   └── Medical_book.pdf
├── research/
├── src/
│   ├── __init__.py
│   ├── agent_graph.py
│   ├── app.py
│   ├── helper.py
│   ├── prompt.py
│   ├── static/
│   └── templates/
│       └── chat.html
└── .vscode/
```

### Key files

- `src/app.py` — Flask application and chat endpoints.
- `src/agent_graph.py` — multi-node LangGraph flow for planning, retrieval, answer generation, and reflection.
- `src/helper.py` — document loading, chunking, and embedding logic.
- `src/prompt.py` — system prompt used by the LLM.
- `store_index.py` — script to ingest PDFs and create Pinecone index records.
- `data/Medical_book.pdf` — disease/medical knowledge source used by the retriever.
- `src/templates/chat.html` — frontend chat UI.

---

## Architecture

The application architecture follows a classical RAG workflow:

1. User enters a medical question in the frontend.
2. Flask receives the request in `src/app.py`.
3. The compiled LangGraph agent executes the workflow from `src/agent_graph.py`.
4. The planner checks whether the input looks like a medical query.
5. If needed, the retriever calls Pinecone to fetch the top relevant document chunks.
6. The answer generator builds a prompt using the system prompt and retrieved context.
7. Gemini generates a final answer.
8. The reflector asks the model to validate whether the answer directly addresses the question.
9. The frontend renders the final answer along with a reflection section.

### Agent flow in the code

The graph is created in `create_medibot_agent()`:

- `planner` decides if retrieval is needed.
- `retriever_node` fetches document context.
- `answer_generator` produces the answer.
- `reflector` validates the answer.
- `END` closes the workflow.

This is implemented using `StateGraph` from `langgraph` and a custom `AgentState` Pydantic model.

---

## Tech stack

### Core tools and libraries

- Python 3.10+
- Flask
- LangChain
- LangGraph
- Google Gemini (`langchain_google_genai`)
- Pinecone vector database
- Hugging Face sentence-transformers
- PyPDF
- python-dotenv

### Example dependencies from `requirements.txt`

```txt
langchain==0.3.26
flask==3.1.1
sentence-transformers==4.1.0
pypdf==5.6.1
python-dotenv==1.1.0
langchain-pinecone==0.2.8
langchain-openai==0.3.24
langchain-community==0.3.26
langchain-google-genai
langgraph==0.2.54
gunicorn
```

---

## How the retrieval pipeline works

The document indexing pipeline is defined in `store_index.py` and the helper utilities in `src/helper.py`.

### Process

1. PDF documents are loaded from `data/` using `DirectoryLoader` and `PyPDFLoader`.
2. The raw PDF text is reduced to minimal document metadata (`source` only).
3. Text is chunked using `RecursiveCharacterTextSplitter`.
4. Each chunk is embedded using `HuggingFaceEmbeddings` with:
   - model: `sentence-transformers/all-MiniLM-L6-v2`
5. Embeddings are uploaded to Pinecone.
6. User queries are embedded with the same model and compared against the stored vectors using cosine similarity.
7. The top three matching documents are returned to the LLM as context.

### Important configuration values from the code

- Pinecone index name: `medibot`
- Retriever top-k: `3`
- Embedding model: `all-MiniLM-L6-v2`
- Vector dimension: `384`
- Metric: `cosine`

This matches the code in `store_index.py` and `src/agent_graph.py`.

---

## Environment setup

### 1. Clone the repository

```bash
git clone https://github.com/codewithkaran-21/MedicalBot-CI-CD.git
cd MedicalBot-CI-CD
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# or
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
PINECONE_INDEX_NAME=medibot
```

The code reads these variables with `python-dotenv` and then sets them into `os.environ` before creating the retriever and LLM.

---

## Data indexing

The repository includes a sample medical PDF in `data/Medical_book.pdf` and a helper to build the Pinecone vector index.

### Run the indexing script

```bash
python store_index.py
```

### What it does

- loads all PDFs in `data/`
- filters documents down to a minimal structure
- splits text into chunks
- creates or reuses the Pinecone index `medibot`
- uploads the embedded text chunks to Pinecone

### Pinecone index creation notes

The code uses:

```python
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

Make sure your Pinecone plan supports this index configuration and your environment is compatible with the selected region.

---

## Running the application

### Local development run

```bash
python src/app.py
```

This starts the Flask app and exposes the chatbot interface on:

```text
http://0.0.0.0:8080
```

### Production-like run

The repository includes a `Procfile`:

```procfile
web: gunicorn src.app:app
```

You can deploy it in a platform that supports Gunicorn, such as Render, Heroku, or similar providers.

For Render, use `pip install -r requirements.txt` as the build command and ensure
`GOOGLE_API_KEY` and `PINECONE_API_KEY` are configured as environment variables.
The Pinecone `medibot` index must already exist before the web service starts.

---

## Web interface

The application serves a simple chat UI from `src/templates/chat.html`.

### Routes

- `/` → home page with chat window
- `/get` → receives message and returns response HTML

### Request flow

When the user sends a message:

```python
result = agent.invoke({"input": msg})
answer = result["answer"]
reflection = result.get("reflection", "")
```

The result is then formatted into HTML and returned to the browser.

---

## Prompt and answer generation

The system prompt is stored in `src/prompt.py`:

```python
system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the answer is not present in the context, use your own reliable medical knowledge. "
    "Answer in three sentences maximum and keep it concise."
    "\n\n"
    "{context}"
)
```

This means the model is instructed to be concise and use the retrieved context, while falling back to general medical knowledge when necessary.

---

## Code-level highlights

### `src/agent_graph.py`

This is the heart of the application. It contains:

- `AgentState` definition
- `plan_node()` for retrieval decisions
- `retrieve_node()` for vector search
- `answer_node()` for LLM response generation
- `reflect_node()` for validating the answer
- `create_medibot_agent()` to compile the graph

### `src/helper.py`

This file contains the helper functions for:

- loading PDFs with `PyPDFLoader`
- splitting documents with `RecursiveCharacterTextSplitter`
- downloading embeddings from Hugging Face
- preparing minimal document metadata for indexing

### `src/app.py`

This file contains:

- Flask app initialization
- environment loading via `dotenv`
- chat route logic
- answer formatting for HTML rendering

---

## Current limitations

This is a functional prototype, but it has some limitations you should be aware of:

1. The planner uses a simple keyword match heuristic rather than a true medical intent classifier.
2. The system does not provide a formal medical-risk policy or a guardrail layer.
3. It is not designed for regulated medical diagnostics or patient-facing clinical decision support.
4. The reflection step is lightweight and may not catch all factual or safety issues.
5. The app assumes a valid Pinecone index and a configured Google API key.

This project is best viewed as a demonstration or research prototype, not a deployable clinical assistant without further validation and safety controls.

---

## Recommended next improvements

- Replace keyword-based planner logic with a stronger intent detection model.
- Add support for multiple knowledge sources beyond PDFs.
- Add logging, metrics, and observability.
- Add user authentication for production deployments.
- Add a proper backend API separate from the HTML UI.
- Improve answer quality with citations to retrieved sources.
- Add disclaimers and risk guardrails for health-related content.
- Add tests for indexing, graph flow, and app routing.

---

## Security and compliance notes

Because the project handles medical questions, it is important to handle data carefully:

- Do not commit API keys or `.env` files.
- Use secure environment management in production.
- Avoid sending personal health records to third-party systems without reviewing privacy policies and legal constraints.
- Any real deployment should include stronger validation, monitoring, and regulatory review.

---

## License

This project includes a `LICENSE` file. Please refer to it for terms of use and distribution rights.

---

## Summary

MedicalBot Agent is a lightweight RAG-powered medical chatbot that uses:

- PDF-based medical data as the knowledge source,
- sentence-transformers for embeddings,
- Pinecone for vector retrieval,
- LangGraph for orchestration,
- Gemini for answer generation,
- Flask for the user interface.

It is a strong starter project for demonstrating AI-driven medical question answering and serves as a practical foundation for building a more robust healthcare assistant.

---

## Quick command reference

```bash
# install deps
pip install -r requirements.txt

# configure env
echo "GOOGLE_API_KEY=..." > .env
echo "PINECONE_API_KEY=..." >> .env
echo "PINECONE_ENV=..." >> .env

# create vector index from PDF
python store_index.py

# run app
python src/app.py
```

---

## Helpful note for deployment

If you plan to deploy this project to a cloud platform:

- secure your environment variables,
- set `gunicorn` in the process manager,
- make sure the Pinecone index exists before app startup,
- verify that the embedding model can download successfully in the runtime environment,
- ensure all API credentials are available at runtime.

---

## Final note

This repository is a useful educational and prototype project for understanding how to combine medical document retrieval with LLM-based conversational systems. It demonstrates the full RAG loop in a compact and accessible form.

If you want to extend it further, the best next steps are adding source attribution, stronger medical safety checks, and a more robust backend architecture.
