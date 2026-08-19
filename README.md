# Nepali AI Chatbot

An AI-powered Nepali conversational system built using Retrieval-Augmented Generation (RAG) to provide accurate and context-aware answers from a structured Nepali knowledge base.

The system combines topic detection, BM25 keyword retrieval, FAISS semantic search, and LLM-based generation to retrieve relevant information before generating the final response.

## 🌐 Deployment

- Frontend: React + Vite — Vercel
- Backend: FastAPI — Render
- Database: PostgreSQL — Render
- Search Index Storage: Cloudflare R2
- LLM: Groq API

---

## 🏗️ System Architecture

                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ React Frontend   │
                         │     Vercel       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   FastAPI API    │
                         │     Render       │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            ┌───────────────┐           ┌───────────────┐
            │ Topic         │           │ Authentication │
            │ Detection     │           │     / JWT      │
            └───────┬───────┘           └───────────────┘
                    │
                    ▼
             ┌──────────────┐
             │ Hybrid Search│
             └──────┬───────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   ┌─────────────┐     ┌─────────────┐
   │    BM25     │     │    FAISS    │
   │  Retrieval  │     │  Retrieval  │
   └──────┬──────┘     └──────┬──────┘
          │                   │
          └─────────┬─────────┘
                    ▼
             ┌──────────────┐
             │ Top Relevant │
             │   Context    │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │   Groq LLM   │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │ Final Answer │
             └──────────────┘

---

## ✨ Features

- 🇳🇵 Nepali language question answering
- Retrieval-Augmented Generation (RAG)
- Hybrid retrieval using BM25 and FAISS
- Topic-based document retrieval
- Semantic similarity search
- Global fallback retrieval
- LLM-powered response generation
- JWT-based authentication
- PostgreSQL database integration
- Cloudflare R2-based index storage
- RESTful API with FastAPI
- Separate frontend and backend deployment
- Production deployment using Vercel and Render

---

## 🧠 RAG Pipeline

### 1. User Query

The user submits a question through the React frontend.

Example:

आगामी आर्थिक वर्षको लागि अनुमान गरिएको कुल सरकारी खर्च कति हो?

### 2. Topic Detection

The query is analyzed to determine the most relevant knowledge category.

The current knowledge base includes:

- अर्थतन्त्र
- पूर्वाधार
- उद्योग र व्यापार
- सामाजिक सुरक्षा
- कृषि
- विज्ञान तथा प्रविधि
- पर्यटन
- शिक्षा
- स्वास्थ्य

If a relevant topic cannot be identified, the system falls back to the global indexes.

### 3. Hybrid Retrieval

The system performs two types of retrieval.

#### BM25

BM25 provides lexical and keyword-based retrieval. It is useful when important terms from the user query directly appear in the indexed documents.

#### FAISS

FAISS performs semantic similarity search using vector embeddings to identify documents that are conceptually similar to the user's query.

### 4. Score Combination

The BM25 and FAISS scores are normalized and combined using:

Hybrid Score = α × BM25 + (1 - α) × FAISS

The highest-ranked documents are selected as the final context.

### 5. LLM Generation

The retrieved context is passed to the Groq-hosted LLM, which generates the final response based on the retrieved information.

---

## 🛠️ Technology Stack

### Frontend

- React
- Vite
- React Router
- Axios
- React Icons
- React Hot Toast

Deployment: Vercel

### Backend

- Python
- FastAPI
- Uvicorn
- SQLAlchemy
- Pydantic
- Alembic

Deployment: Render

### AI & Retrieval

- Sentence Transformers
- FAISS
- BM25
- NumPy
- scikit-learn
- Groq
- LangChain

### Database

- PostgreSQL
- SQLAlchemy Async
- asyncpg

Hosting: Render

### Storage

- Cloudflare R2
- boto3

FAISS and BM25 indexes are stored in Cloudflare R2 and downloaded by the backend when required.

### Authentication

- JWT
- python-jose
- bcrypt
- passlib

---

## 📁 Project Structure

project/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── loader.py
│   │   │
│   │   ├── embeddings/
│   │   │   └── model_embeddings.py
│   │   │
│   │   ├── retriever/
│   │   │   ├── hybrid_search.py
│   │   │   └── topic_detect.py
│   │   │
│   │   └── ...
│   │
│   ├── scripts/
│   │   └── download_indexes.py
│   │
│   ├── requirements.txt
│   ├── alembic.ini
│   └── .env
│
└── frontend/
    ├── src/
    ├── public/
    ├── package.json
    └── ...

---

## ⚙️ Local Development

### Backend Setup

Clone the repository:

git clone <repository-url>
cd <repository-name>

Create a virtual environment:

python -m venv venv

Activate the virtual environment.

Windows:

venv\Scripts\activate

Linux/macOS:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Configure the required environment variables in `.env`.

Download the search indexes:

python -m scripts.download_indexes

Start the FastAPI development server:

uvicorn app.main:app --reload

The backend will be available at:

http://localhost:8000

Interactive API documentation:

http://localhost:8000/docs

---

## 💻 Frontend Setup

Navigate to the frontend directory:

cd frontend

Install dependencies:

npm install

Start the development server:

npm run dev

The frontend will be available at the local Vite development URL displayed in the terminal.

---

## 🔐 Environment Variables

The application requires environment variables for database access, authentication, LLM services, object storage, and embedding services.

Example:

DATABASE_URL=your_database_url

SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

GROQ_API_KEY=your_groq_api_key

R2_ACCOUNT_ID=your_r2_account_id
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=your_bucket_name
R2_ENDPOINT=your_r2_endpoint

HF_TOKEN=your_huggingface_token

Replace the values with your actual credentials.

Never commit API keys, database credentials, tokens, or `.env` files to the repository.

---

## 🚀 Deployment

The application uses a separated deployment architecture:

| Component | Technology | Deployment |
|-----------|------------|------------|
| Frontend | React + Vite | Vercel |
| Backend | FastAPI | Render |
| Database | PostgreSQL | Render |
| Search Index Storage | Cloudflare R2 | Cloudflare |
| LLM | Groq | Groq API |

### Backend Deployment

The backend is deployed on Render.

Production start command:

python -m scripts.download_indexes && uvicorn app.main:app --host 0.0.0.0 --port $PORT

During deployment, the application:

1. Installs the required Python dependencies.
2. Downloads FAISS and BM25 indexes from Cloudflare R2.
3. Loads the search indexes.
4. Starts the FastAPI application using Uvicorn.
5. Exposes the API through Render.

### Frontend Deployment

The React frontend is deployed on Vercel.

The frontend communicates with the deployed FastAPI backend through the configured production API URL.

### Database

The application uses PostgreSQL hosted on Render for persistent application data.

Database operations are handled using:

- SQLAlchemy
- asyncpg
- Alembic

---

## ☁️ Cloudflare R2 Index Storage

The FAISS and BM25 search indexes are stored in Cloudflare R2.

The repository does not need to contain the complete search index collection.

At deployment/startup, the backend downloads the required indexes using:

python -m scripts.download_indexes

This keeps the source repository lightweight while allowing the search indexes to be managed independently.

---

## 🔎 Example Query Workflow

Example user query:

शिक्षा क्षेत्रमा सरकारले कति बजेट छुट्याएको छ?

The system processes the request as follows:

User Query
    │
    ▼
Topic Detection
    │
    ▼
शिक्षा
    │
    ├───────────────┐
    ▼               ▼
  BM25            FAISS
 Search           Search
    │               │
    └───────┬───────┘
            ▼
      Hybrid Ranking
            │
            ▼
   Relevant Documents
            │
            ▼
        Groq LLM
            │
            ▼
      Final Nepali Answer

---

## 🔐 Authentication

The application supports JWT-based authentication.

Authentication flow:

User Registration
       │
       ▼
Password Hashing
       │
       ▼
PostgreSQL
       │
       ▼
User Login
       │
       ▼
JWT Access Token
       │
       ▼
Authenticated Requests

Passwords are hashed before being stored in the database.

---

## 📊 Hybrid Search

The hybrid retriever combines lexical and semantic retrieval.

Default weighting:

alpha = 0.5

Therefore:

50% BM25
+
50% FAISS

The weighting can be adjusted based on retrieval performance and evaluation results.

---

## 🧪 Testing

Start the backend locally:

uvicorn app.main:app --reload

Open the Swagger API documentation:

http://localhost:8000/docs

Swagger UI can be used to test the available API endpoints.

---

## 🔒 Security

Sensitive credentials are managed through environment variables.

The following should never be committed to GitHub:

- `.env`
- API Keys
- Database Credentials
- JWT Secret Keys
- Cloudflare R2 Credentials
- Hugging Face Tokens

Recommended `.gitignore`:

.env
venv/
__pycache__/
*.pyc
node_modules/
dist/

---

## 🔮 Future Improvements

- Improved Nepali semantic retrieval
- Retrieval evaluation and benchmarking
- Reranking models
- Conversation history and memory
- Streaming LLM responses
- Query and embedding caching
- Expanded Nepali knowledge base
- Improved response evaluation
- Performance and memory optimization
- Application monitoring and analytics

---

## 👨‍💻 Author

**Sandip Bhujel**

Computer Engineering | Backend & AI Developer

---
