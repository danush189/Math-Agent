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
## Usage
1. **Ingest Knowledge Base:**
   - Run `python ingest_math_knowledge_base.py` to populate Qdrant with math problems.
2. **Query the Agent:**
   - Run `python math_agent_query.py` and enter a math question to retrieve similar problems and solutions.

## Next Steps (Completed)
- Agent workflow/orchestration (LangGraph)
- Web search fallback (Tavily API) for out-of-domain queries
- Human-in-the-loop feedback (DSPy)
- Web interface (Chainlit)

## Possible Future Work
- Add advanced features (MCP, JEE Bench evaluation, etc)

---
_This README will be updated as the project progresses._
