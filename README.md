# 🛡️ Toxic Guard: AI-Powered Poisoning Triage System

Toxic Guard is an AI-assisted clinical decision support tool designed to analyze poisoning and toxicology cases. Built with a Retrieval-Augmented Generation (RAG) architecture, it cross-references patient symptoms against localized medical documents and live web data to provide explainable triage recommendations.

**⚠️ MEDICAL DISCLAIMER:** *This application is a proof-of-concept for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult emergency services (e.g., 911 or Poison Control) in real-world emergencies.*

---

## ✨ Key Features
* **Hybrid RAG Architecture:** Combines offline document retrieval (PDFs) with live web search using Cohere's connectors.
* **Deterministic Safety Layer:** A hardcoded, rule-based triage system that immediately flags "CRITICAL" or "HIGH RISK" keywords (e.g., seizures, unconsciousness) before the LLM processes the query.
* **Explainable AI:** Generates step-by-step reasoning for its clinical recommendations, ensuring transparency for healthcare providers.
* **Unified Streamlit Interface:** A clean, responsive, single-page application that handles both the UI and the backend orchestration.

---

## 🛠️ Tech Stack
* **Frontend & Orchestration:** [Streamlit](https://streamlit.io/)
* **LLM & Embeddings:** [Cohere](https://cohere.com/) (`command-a-03-2025` and `embed-english-v3.0`)
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (CPU)
* **Document Processing:** `LangChain` (Text Splitting) & `PyPDF`

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/toxic-guard.git](https://github.com/yourusername/toxic-guard.git)
cd toxic-guard
