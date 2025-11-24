# RAG & Tools Assistant — Multi-Modal AI Agent

A production-ready AI agent that can decide *how* to answer — whether by searching internal documents, calling an external API, or responding directly using reasoning. No over-engineering, no magic — just clean routing logic powered by **LangGraph** and **Google Gemini**.

---

## What This Agent Does

This system acts like an experienced researcher:

- If the user asks about **weather**, it reaches out to OpenWeatherMap.
- If the user asks something covered in the uploaded **PDF**, it performs retrieval via Qdrant and returns grounded context.
- If neither is needed, it responds directly using the LLM.

The result: fast answers, minimal hallucination, and zero unnecessary API calls.

---

## Architecture Overview

User Query → Router  
    ├── Weather → OpenWeatherMap API  
    ├── Document → RAG Pipeline (Qdrant + PDF)  
    └── General → Direct LLM Response

| Component | Responsibility |
|----------|----------------|
| graph_agent.py | Routing logic & state transitions |
| tools/find_weather.py | Weather tool wrapper |
| tools/rag.py | Ingestion, embedding, retrieval |
| main.py | Streamlit chat interface |

---

## Quickstart

git clone <repo-url>
cd ai-agent-assignment
pip install -r requirements.txt
cp .env.example .env # Add your API keys
streamlit run main.py

yaml
Copy code

You’ll need valid keys for **Gemini** and **OpenWeatherMap**.

---

## Example Queries & Behaviors

| Query | What Happens |
|-------|--------------|
| "What’s the weather in Tokyo?" | Weather tool call |
| "Explain one-shot prompting" | RAG search on PDF |
| "Hi, who are you?" | Direct LLM reply |

---

## Testing

Unit tests & Integration tests (real tool calls)
python tests/test_agent.py

java
Copy code

For LangSmith evaluation (10 samples):

python eval_agent.py

yaml
Copy code

---

## 🧰 Configuration

`.env` should include:

GOOGLE_API_KEY=...
OPENWEATHERMAP_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true

yaml
Copy code


---

## 🗂 Project Layout

.
├── graph_agent.py
├── main.py
├── requirements.txt
├── .env
├── resources/
│ └── AI_Agents.pdf
├── tools/
│ ├── find_weather.py
│ └── rag.py
└── tests/
├── test_agent.py
└── test_integration.py

yaml
Copy code

---
