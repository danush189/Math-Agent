# Mathematical AI Agent System: Project README

## Project Overview
This project implements a Human-in-the-Loop Mathematical Professor AI Agent. It leverages a vector database (Qdrant) for mathematical knowledge storage and retrieval, and supports advanced agent workflows for solving and explaining math problems.

## Progress & Steps

### Step 1: Vector Database & Knowledge Base Ingestion
- Set up Qdrant (vector database) using Docker or native install.
- Ingested two datasets:
  - **GSM8K**: Grade school math problems with step-by-step solutions.
  - **MathQA**: Word problems with operational programs and rationales.
- Combined both datasets, generated embeddings using `sentence-transformers`, and uploaded to Qdrant with topic and difficulty metadata.
- Script: `ingest_math_knowledge_base.py`

### Step 2: Agent Query Logic
- Created a script to query the vector database for the most similar math problems and solutions.
- Returns top-k closest questions, answers, topics, and difficulty levels for a user query.
- Script: `math_agent_query.py`

### Step 3: Basic Agent Workflow (LangGraph)
- Added a minimal LangGraph workflow in `math_agent_workflow.py`.
- The workflow routes user queries to the knowledge base search node and returns the best match from Qdrant.
- This sets the foundation for adding more advanced agent logic, fallback, and feedback in future steps.

### Step 4: Web Search Fallback (Tavily)
- Added Tavily web search integration as a fallback node in the agent workflow (`math_agent_workflow.py`).
- API key is loaded from `.env` using the variable `TAVILY_API_KEY`.
- If the knowledge base does not return a confident result, the agent queries Tavily for relevant math solutions from trusted domains.

### Step 5: LLM Reasoning (OpenAI Integration)
- Integrated OpenAI's GPT-3.5-turbo for step-by-step math solution synthesis.
- The agent now uses the knowledge base, falls back to web search if needed, and always synthesizes a final answer using the LLM.
- Add your OpenAI API key to the `.env` file as `OPENAI_API_KEY`.
- The output now includes the original question, aggregated info from KB/web, and an LLM-generated step-by-step solution.

---

## Detailed Progress Update (as of July 2025)

- Qdrant vector DB and ingestion scripts for GSM8K/MathQA datasets set up.
- Implemented retrieval, web search fallback (Tavily), and LLM synthesis in `math_agent_workflow.py`.
- Integrated OpenAI LLM for LaTeX-formatted math solutions.
- Built Chainlit UI (`math_agent_chainlit.py`) for Q&A, feedback, and DSPy prompt optimization.
- Added feedback logging and DSPy-compatible dataset creation.
- Automated DSPy prompt optimization after every 5 feedbacks.
- Added agentic verification and revision loop to the workflow.
- Improved math formatting in Chainlit UI: only wrap equations in `$$...$$`, fix LaTeX symbols, and bold final answers.
- Updated LLM prompt to enforce concise, step-by-step, block-LaTeX output, and later allowed for long, detailed solutions.
- Expanded Tavily web search to include a broad set of authoritative math domains.
- Fixed feedback UI so it appears for every response, not just the first.
- Updated `.gitignore` to exclude feedback logs, DSPy logs, Chainlit logs, `.env`, and other sensitive/generated files.
- Confirmed that only specific log/training JSON files are ignored, not all JSON files.
- Refactored DSPy training to use `BootstrapFewShotWithRandomSearch`.
- Updated Chainlit UI to display when a DSPy-optimized prompt is in use.
- Fixed import errors for DSPy’s `Predict`.
- Overhauled math formatting: only wrap equations in `$$...$$`, fix LaTeX symbols, bold final answers, and avoid wrapping text.
- Updated LLM prompt for concise, block-LaTeX output, then allowed for long, detailed solutions.
- Expanded Tavily web search fallback to include many more math-related domains for better coverage.
- Improved feedback logging and DSPy dataset creation.
- Ensured all code changes are compatible with the latest DSPy and LangChain APIs.
- Reset `feedback_given` in Chainlit UI for every new question so feedback options always appear.
- If secrets were ever committed, use `git filter-repo` or BFG to remove them from history and force-push after cleaning.

## Usage
1. **Ingest Knowledge Base:**
   - Run `python ingest_math_knowledge_base.py` to populate Qdrant with math problems.
2. **Query the Agent:**
   - Run `python math_agent_query.py` and enter a math question to retrieve similar problems and solutions.

## Next Steps
- Integrate agent workflow/orchestration (LangGraph).
- Add web search fallback (Tavily API) for out-of-domain queries.
- Implement human-in-the-loop feedback (DSPy).
- Build a web interface (Chainlit).
- Add advanced features (MCP, JEE Bench evaluation, etc).

---
_This README will be updated as the project progresses._
