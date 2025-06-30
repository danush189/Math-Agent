# math_agent_workflow.py
# Minimal LangGraph workflow for math agent knowledge base search

from langgraph.graph import StateGraph, END
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import requests
import openai

# Define agent state
class MathAgentState(TypedDict):
    question: str
    result: dict

# Node: Search knowledge base
def search_knowledge_base(state: MathAgentState) -> MathAgentState:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = QdrantClient("localhost", port=6333)
    collection_name = "math_problems"
    query_embedding = model.encode([state["question"]])[0]
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=1
    )
    if results:
        r = results[0]
        state["result"] = {
            "question": r.payload["question"],
            "answer": r.payload["answer"],
            "topic": r.payload.get("topic", "unknown"),
            "difficulty": r.payload.get("difficulty", "unknown"),
            "score": r.score
        }
    else:
        state["result"] = None
    return state

# Node: Web search fallback
def web_search_node(state: MathAgentState) -> MathAgentState:
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = TavilyClient(api_key=api_key)
    response = tavily_client.search(
        query=f"mathematical solution step by step {state['question']}",
        search_depth="advanced",
        max_results=5,  # get more results to filter for relevance
        include_domains=[
            "khanacademy.org", "mathstackexchange.com", "mathoverflow.net", "wolfram.com",
            "brilliant.org", "artofproblemsolving.com", "mathisfun.com", "purplemath.com",
            "symbolab.com", "desmos.com", "byjus.com", "cuemath.com", "mathhelp.com",
            "mathway.com", "chegg.com", "socratic.org", "edx.org", "coursera.org",
            "openstax.org", "mit.edu", "stanford.edu", "harvard.edu"
        ]
    )
    # Filter results for high relevance to the question
    def is_relevant(result):
        content = result.get('content', '').lower()
        question = state['question'].lower()
        # Consider relevant if at least half the question words appear in the content
        qwords = [w for w in question.split() if len(w) > 2]
        match_count = sum(1 for w in qwords if w in content)
        return match_count >= max(1, len(qwords) // 2)
    if response and response.get('results'):
        filtered = [r for r in response['results'] if is_relevant(r)]
        results = filtered[:1] if filtered else response['results'][:1]  # take the most relevant
        answer = "\n\n".join([
            f"Source: {r.get('url', 'N/A')}\nContent: {r.get('content', 'No answer found.').strip()}"
            for r in results
        ])
        state['result'] = {
            'question': state['question'],
            'answer': answer,
            'topic': 'web_search',
            'difficulty': 'unknown',
            'score': 0.0
        }
    else:
        state['result'] = None
    return state

def llm_synthesis_node(state: MathAgentState) -> MathAgentState:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        state['result']['llm_answer'] = "[LLM API key not set. Please add OPENAI_API_KEY to your .env file.]"
        return state
    feedback_note = state.get('feedback_note', '')
    # Use DSPy optimized prompt if available
    optimized_prompt = state.get('optimized_prompt')
    if optimized_prompt:
        prompt = optimized_prompt.format(
            question=state['question'],
            retrieved_info=state['result']['answer'],
            feedback_note=feedback_note
        )
    else:
        prompt = f"""
You are a helpful math tutor. Given the following math question and retrieved information, provide a clear, step-by-step solution.

For each step, use a short explanation (if needed), then put the equation on its own line, wrapped in $$ ... $$. Use LaTeX for all math symbols and expressions. Do NOT use $...$ for entire sentences or explanations.

If you use square roots, write them as \\sqrt{{...}}. For exponents, use ^. For fractions, use \\frac{{a}}{{b}}.

Be concise and only show the essential steps and final answer. If the solution requires more steps, do not omit any important step.

Question: {state['question']}

Retrieved Information:
{state['result']['answer']}

Step-by-step solution:
"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,  # Allow long responses
            temperature=0.2,
            top_p=0.95
        )
        llm_answer = response.choices[0].message.content.strip()
    except Exception as e:
        llm_answer = f"[LLM error: {e}]"
    state['result']['llm_answer'] = llm_answer
    return state

def verification_agent(question, solver_answer):
    """
    Verification agent: 1) attempts to solve the question independently, 2) tries to backtrack from the provided answer to the question.
    Returns a dict with 'independent_solution', 'backtrack_check', and 'verdict'.
    """
    import openai
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    # 1. Independent solution
    prompt_independent = f"""
You are a math professor. Solve the following question step by step, as if you have not seen any previous answer. Format all math in LaTeX.

Question: {question}

Step-by-step solution:
"""
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_independent}],
        max_tokens=512,
        temperature=0.2
    )
    independent_solution = response1.choices[0].message.content.strip()
    # 2. Backtrack check
    prompt_backtrack = f"""
You are a math professor. Given the following answer, try to reconstruct the original question or check if the answer is a valid solution to the question. Explain your reasoning step by step.

Question: {question}

Provided Answer: {solver_answer}

Backtrack/Validation:
"""
    response2 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_backtrack}],
        max_tokens=512,
        temperature=0.2
    )
    backtrack_check = response2.choices[0].message.content.strip()
    # Simple verdict (could be improved with LLM or logic)
    verdict = "Verified" if "correct" in backtrack_check.lower() or "valid" in backtrack_check.lower() else "Needs Review"
    return {
        "independent_solution": independent_solution,
        "backtrack_check": backtrack_check,
        "verdict": verdict
    }

# Build workflow
def create_math_agent():
    """
    Returns a compiled workflow agent that accepts a state dict with keys:
      - question: str
      - result: dict (optional)
      - feedback_note: str (optional)
      - optimized_prompt: str (optional, for DSPy prompt injection)
    """
    workflow = StateGraph(MathAgentState)
    workflow.add_node("knowledge_search", search_knowledge_base)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("llm_synthesis", llm_synthesis_node)
    def route(state: MathAgentState):
        if not state["result"] or state["result"].get("score", 0) < 0.8:
            return "web_search"
        return "llm_synthesis"
    def after_web_search(state: MathAgentState):
        return "llm_synthesis"
    workflow.set_entry_point("knowledge_search")
    workflow.add_conditional_edges("knowledge_search", route, {"web_search": "web_search", "llm_synthesis": "llm_synthesis"})
    workflow.add_conditional_edges("web_search", after_web_search, {"llm_synthesis": "llm_synthesis"})
    workflow.add_edge("llm_synthesis", END)
    return workflow.compile()

# Main workflow with verification, now supports feedback_note and optimized_prompt

def main_workflow_with_verification(question, feedback_note=None, optimized_prompt=None):
    """
    Main workflow: 1) Retrieve or search, 2) Solve, 3) Verify, 4) Return result and verification.
    Accepts feedback_note and optimized_prompt for DSPy integration.
    """
    # Step 1: Build initial state
    state = {"question": question, "result": None}
    if feedback_note:
        state["feedback_note"] = feedback_note
    if optimized_prompt:
        state["optimized_prompt"] = optimized_prompt
    agent = create_math_agent()
    result_state = agent.invoke(state)
    result = result_state["result"]
    if not result:
        return {"error": "No similar problem found."}
    solver_answer = result.get("llm_answer", "[No LLM answer]")
    # Step 2: Verification agent
    verification = verification_agent(question, solver_answer)
    # Step 3: If verification fails, try to revise the solution once
    if verification["verdict"] != "Verified":
        # Try to revise the solution using the feedback from the verification agent
        revision_prompt = f"""
You are a math professor. The following solution was flagged as needing review. Here is the original question and the previous answer. Please revise the answer to address any issues and provide a clear, correct, step-by-step solution in LaTeX.

Question: {question}

Previous Answer: {solver_answer}

Verification Feedback: {verification['backtrack_check']}

Revised Step-by-step Solution:
"""
        import openai
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": revision_prompt}],
            max_tokens=512,
            temperature=0.2
        )
        revised_solution = response.choices[0].message.content.strip()
        # Optionally, re-verify the revised solution
        verification2 = verification_agent(question, revised_solution)
        return {
            "question": result["question"],
            "aggregated_info": result["answer"].strip(),
            "llm_solution": solver_answer,
            "topic": result["topic"],
            "difficulty": result["difficulty"],
            "score": result["score"],
            "verification": verification,
            "revised_solution": revised_solution,
            "revised_verification": verification2
        }
    # Step 4: Return both
    return {
        "question": result["question"],
        "aggregated_info": result["answer"].strip(),
        "llm_solution": solver_answer,
        "topic": result["topic"],
        "difficulty": result["difficulty"],
        "score": result["score"],
        "verification": verification
    }

if __name__ == "__main__":
    user_query = input("Enter your math question: ")
    output = main_workflow_with_verification(user_query)
    if "error" in output:
        print(output["error"])
    else:
        print(f"\nBest match:")
        print(f"Question: {output['question']}")
        print(f"Aggregated Info:\n{output['aggregated_info']}")
        print(f"LLM Step-by-step Solution:\n{output['llm_solution']}")
        print(f"Topic: {output['topic']}")
        print(f"Difficulty: {output['difficulty']}")
        print(f"Score: {output['score']:.2f}")
        print("\n--- Verification Agent ---")
        print(f"Independent Solution:\n{output['verification']['independent_solution']}")
        print(f"Backtrack/Validation:\n{output['verification']['backtrack_check']}")
        print(f"Verdict: {output['verification']['verdict']}")
        if 'revised_solution' in output:
            print("\n--- Revised Solution (after failed verification) ---")
            print(output['revised_solution'])
            print("\n--- Revised Verification ---")
            print(f"Independent Solution:\n{output['revised_verification']['independent_solution']}")
            print(f"Backtrack/Validation:\n{output['revised_verification']['backtrack_check']}")
            print(f"Verdict: {output['revised_verification']['verdict']}")
