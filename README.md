# 🛡️ Toxic Guard: AI-Powered Poisoning Triage System

**Toxic Guard** is an advanced, AI-assisted clinical decision support tool designed to analyze toxicology and poisoning cases in real-time. By leveraging a Hybrid Retrieval-Augmented Generation (RAG) architecture, it cross-references patient symptoms against localized medical documents and live web data to provide rapid, explainable triage recommendations.

---

## ⚠️ MEDICAL DISCLAIMER
**This application is a proof-of-concept designed for educational, research, and demonstration purposes only.** It is **not** a substitute for professional medical advice, diagnosis, or treatment. The AI-generated recommendations should never override human clinical judgment. In a real-world emergency, always consult emergency services (e.g., 911) or a certified Poison Control Center.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Tech Stack](#-tech-stack)
5. [Installation & Setup](#-installation--setup)
6. [Usage Guide](#-usage-guide)
7. [Project Structure](#-project-structure)
8. [Testing](#-testing)
9. [Author](#-author)

---

## 🔍 Project Overview
When responding to potential poisoning cases, healthcare professionals need rapid access to specific toxicology protocols. Toxic Guard streamlines this process by ingesting a user-provided case description and passing it through a multi-layered analysis pipeline. It evaluates immediate life threats using a deterministic safety layer, fetches relevant medical literature from a local vector database, and queries the web for the latest updates—synthesizing all this into a clear, actionable triage report.

---

## ✨ Key Features
* **Deterministic Safety Layer:** A hardcoded rule-based system that immediately intercepts the query to scan for critical medical red flags (e.g., "seizure", "unconscious", "cardiac arrest") before any LLM processing occurs, ensuring life-threatening symptoms are never missed.
* **Hybrid RAG Architecture:** * *Offline Retrieval:* Uses FAISS to perform similarity searches against a local vector database containing medical guidelines (PDF).
  * *Live Web Retrieval:* Utilizes Cohere's web connectors to fetch the latest, publicly known toxicology updates.
* **Explainable AI Outputs:** The system doesn't just give a risk level; it provides a step-by-step clinical analysis, citing exactly why a recommendation was made based on the retrieved evidence.
* **Unified Streamlit Interface:** A clean, responsive frontend that handles both user interaction and backend orchestration without the need for a separate API server.

---

## ⚙️ System Architecture
1. **Input:** User submits a patient case description via the Streamlit UI.
2. **Safety Triage:** `safety_layer.py` scans the text. If critical keywords are found, it instantly flags the risk as `CRITICAL` or `HIGH`.
3. **Document Ingestion (Cached):** On startup, `rag_engine.py` extracts text from `data/toxicology_study.pdf`, chunks it using LangChain, embeds it using Cohere, and stores it in a FAISS index.
4. **Hybrid Retrieval:** The system embeds the user's query, retrieves the top `k` most relevant chunks from FAISS, and simultaneously queries the web.
5. **Generation:** Cohere's `command-a-03-2025` model synthesizes the safety flag, the document context, and the web context to generate the final, formatted triage report.

---

## 🛠️ Tech Stack
* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **Large Language Model (LLM):** [Cohere](https://cohere.com/) (`command-a-03-2025`)
* **Embedding Model:** Cohere (`embed-english-v3.0`)
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (CPU-optimized)
* **Document Parsing:** `PyPDF`
* **Text Processing:** `LangChain` (RecursiveCharacterTextSplitter)
* **Environment Management:** `python-dotenv`

---

## 🚀 Installation & Setup

### 1. Prerequisites
* Python 3.9 or higher installed on your machine.
* A valid [Cohere API Key](https://dashboard.cohere.com/api-keys).

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/toxic-guard.git](https://github.com/yourusername/toxic-guard.git)
cd toxic-guard
```

### 3. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install streamlit cohere faiss-cpu numpy pypdf langchain-text-splitters python-dotenv
```

### 5. Configure Environment Variables
Create a file named exactly `.env` in the root directory of the project. Add your Cohere API key:
```env
COHERE_API_KEY=your_actual_api_key_here
```
*(Ensure there are no spaces around the `=` sign and no quotation marks around the key).*

### 6. Add Toxicology Data
1. Create a folder named `data` in the root directory.
2. Place your reference medical document inside the folder and name it exactly `toxicology_study.pdf`.

---

## 🖥️ Usage Guide

To start the application, run the following command in your terminal (ensure your virtual environment is activated):
```bash
streamlit run app.py
```

1. The application will automatically open in your default web browser at `http://localhost:8501`.
2. On the first run, the system will take a few seconds to parse, chunk, and embed the PDF into the FAISS vector database.
3. Once the **System Status** sidebar shows "✅ Vector Database Loaded", you can begin entering case descriptions.
4. Click **Analyze Case** to view the generated risk level, clinical analysis, and recommended actions.

---

## 📂 Project Structure

```text
toxic-guard/
│
├── data/
│   └── toxicology_study.pdf    # Local medical reference document for RAG
│
├── .env                        # Environment variables (API keys)
├── app.py                      # Main Streamlit frontend and orchestrator
├── config.py                   # Environment variable loader
├── rag_engine.py               # Handles embeddings, FAISS indexing, and LLM calls
├── safety_layer.py             # Deterministic keyword-based triage logic
├── test_scenarios.txt          # Pre-written clinical scenarios for testing
└── README.md                   # Project documentation
```

---

## 🧪 Testing
A `test_scenarios.txt` file is included in the repository. It contains various complex, multi-layered clinical scenarios (e.g., mixed ingestions, pediatric exposures, occupational hazards). You can copy and paste these into the Streamlit UI to test the RAG engine's retrieval accuracy and the Safety Layer's sensitivity.

---

## 👨‍💻 Author
**Faisal Shazad** *AI & Machine Learning Engineer*

---
*Powered by Cohere AI | FAISS Vector Database | RAG Architecture*
```
