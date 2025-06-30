import chainlit as cl
from typing import Optional
from math_agent_workflow import create_math_agent
from chainlit.input_widget import Select
import json
from datetime import datetime
import os
import dspy

agent = create_math_agent()

@cl.on_chat_start
def start():
    cl.user_session.set("agent", agent)
    return cl.Message(
        content="Hello! I'm your Mathematical Professor AI. Ask me any math question and I'll provide step-by-step solutions!"
    )

# Helper to find feedback for a question
async def get_feedback_for_question(question, feedback_file="feedback_log.json"):
    if not os.path.exists(feedback_file):
        return []
    feedbacks = []
    try:
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("question", "").strip().lower() == question.strip().lower():
                    feedbacks.append(entry)
    except Exception as e:
        print(f"[Feedback read error]: {e}")
    return feedbacks

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    user_query = message.content
    # Reset feedback for each new question
    cl.user_session.set("feedback_given", False)
    # Check for past feedback
    feedbacks = await get_feedback_for_question(user_query)
    feedback_note = ""
    if feedbacks:
        negative = [f for f in feedbacks if f["feedback_type"] in ["error", "clarify"]]
        if negative:
            feedback_note = "\\n\\nNote: Previous users reported issues or requested clarification for this question. Please be extra clear, detailed, and double-check your solution."
    # Use DSPy optimized prompt if available
    optimized_prompt = PROMPT_OPTIMIZER.prompt if hasattr(PROMPT_OPTIMIZER, 'prompt') and PROMPT_OPTIMIZER.prompt else None
    state = {"question": user_query, "result": None, "feedback_note": feedback_note, "optimized_prompt": optimized_prompt}
    result_state = agent.invoke(state)
    result = result_state["result"]
    if result:
        # Format LLM output for math: only wrap lines that look like equations in $$...$$
        llm_answer = result.get('llm_answer', '[No LLM answer]')
        import re
        def wrap_equation_lines(text):
            lines = text.split('\n')
            new_lines = []
            for i, line in enumerate(lines):
                # Clean up whitespace
                line = line.strip()
                # If line already has $$, leave it
                if re.match(r'^\$\$.*\$\$$', line):
                    new_lines.append(line)
                # If line looks like a LaTeX equation (contains =, +, -, *, /, ^, \sqrt, \frac)
                elif re.search(r'(=|\\sqrt|\\frac|\^|[0-9]+\\/)', line) and len(line) < 80:
                    # Only wrap if not already inline math
                    if not line.startswith('$$'):
                        new_lines.append(f'$$ {line} $$')
                    else:
                        new_lines.append(line)
                # Bold the final answer if it looks like 'c = ...' or 'Answer:'
                elif re.match(r'^(c\s*=|Answer:)', line, re.IGNORECASE):
                    new_lines.append(f'**{line}**')
                else:
                    new_lines.append(line)
            return '\n'.join(new_lines)
        # Fix sqrt formatting and ensure LaTeX symbols
        llm_answer = re.sub(r'sqrt\{([^}]+)\}', r'\\sqrt{\1}', llm_answer)
        llm_answer = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', llm_answer)
        llm_answer = wrap_equation_lines(llm_answer)
        # Extract sources from the aggregated info if present
        sources = []
        if 'Source:' in result['answer']:
            for line in result['answer'].split('\n'):
                if line.startswith('Source:'):
                    url = line.replace('Source:', '').strip()
                    if url and url != 'N/A':
                        sources.append(url)
        # Only allow feedback once per response
        feedback_given = cl.user_session.get("feedback_given", False)
        if not feedback_given:
            actions = [
                cl.Action(name="rate_solution", value="rate", description="Rate this solution", payload={}),
                cl.Action(name="request_clarification", value="clarify", description="Request clarification", payload={}),
                cl.Action(name="report_error", value="error", description="Report an error", payload={})
            ]
        else:
            actions = []
        # Add a note if an optimized prompt is in use
        prompt_note = "\n\n*Using DSPy-optimized prompt for this answer.*" if optimized_prompt else ""
        # Compose sources section
        sources_section = ""
        if sources:
            sources_section = "\n\n**Relevant Sources:**\n" + "\n".join([f"- [{url}]({url})" for url in sources])
        await cl.Message(
            content=f"**Question:** {result['question']}\n\n**Step-by-step Solution:**\n{llm_answer}{prompt_note}{sources_section}\n\n**Topic:** {result['topic']}\n**Difficulty:** {result['difficulty']}\n**Score:** {result['score']:.2f}",
            actions=actions
        ).send()
        cl.user_session.set("last_question", result['question'])
        cl.user_session.set("last_llm_answer", llm_answer)
    else:
        await cl.Message(content="No similar problem found.").send()

@cl.action_callback("rate_solution")
async def on_rate_solution(action):
    cl.user_session.set("feedback_given", True)
    await log_feedback("rate", action)
    await cl.Message(content="Thank you for your feedback! (Rating received)").send()

@cl.action_callback("request_clarification")
async def on_request_clarification(action):
    cl.user_session.set("feedback_given", True)
    await log_feedback("clarify", action)
    await cl.Message(content="Please specify what you would like clarified.").send()

@cl.action_callback("report_error")
async def on_report_error(action):
    cl.user_session.set("feedback_given", True)
    await log_feedback("error", action)
    await cl.Message(content="Thank you for reporting the error. We will review this solution.").send()

# DSPy: Feedback dataset path
FEEDBACK_DATASET_PATH = "feedback_dspy.jsonl"
FEEDBACK_TRAIN_THRESHOLD = 5

# DSPy: Simple prompt optimizer (example)
class SimplePromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = None
    def forward(self, input):
        # In a real scenario, you would use DSPy APIs to optimize prompt or model
        # Here, just return input for demonstration
        return input

PROMPT_OPTIMIZER = SimplePromptOptimizer()

# Helper to count lines in feedback file
def count_feedback_entries(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# DSPy: Real training function using feedback
from dspy import Example
from dspy.datasets.dataset import Dataset
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
from dspy.predict import Predict

# DSPy: Real train function

def train_with_feedback():
    print("[DSPy] Training with feedback...")
    # Load feedback dataset
    examples = []
    try:
        with open(FEEDBACK_DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                # Only use positive/clarified feedback for training
                if entry.get("feedback") in ["rate", "clarify"]:
                    examples.append(Example(input=entry["input"], output=entry["output"]))
    except Exception as e:
        print(f"[DSPy] Error loading feedback: {e}")
        return
    if not examples:
        print("[DSPy] No suitable feedback for training.")
        return
    # Create DSPy dataset
    dataset = Dataset("feedback-dspy", train=examples, test=[])
    # Use BootstrapFewShotWithRandomSearch for more robust prompt optimization
    teleprompter = BootstrapFewShotWithRandomSearch(metric="exact_match", max_bootstrapped_demos=3, num_candidate_programs=3)
    predictor = Predict("input -> output")
    # Train prompt
    teleprompter.compile(predictor, dataset)
    # Save the optimized prompt for use in the agent (in-memory for now)
    PROMPT_OPTIMIZER.prompt = teleprompter.prompt
    print("[DSPy] Training complete. Optimized prompt:")
    print(PROMPT_OPTIMIZER.prompt)

# Update feedback logging to DSPy-compatible format and trigger training
async def log_feedback(feedback_type, action):
    feedback = {
        "timestamp": datetime.utcnow().isoformat(),
        "feedback_type": feedback_type,
        "question": cl.user_session.get("last_question", ""),
        "llm_answer": cl.user_session.get("last_llm_answer", ""),
        "action_payload": action.payload if hasattr(action, 'payload') else {},
        "user_id": getattr(action, 'user_id', None)
    }
    # Write to both the old log and DSPy dataset
    try:
        with open("feedback_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback) + "\n")
        # Write DSPy-compatible feedback
        dspy_feedback = {
            "input": feedback["question"],
            "output": feedback["llm_answer"],
            "feedback": feedback["feedback_type"]
        }
        with open(FEEDBACK_DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(dspy_feedback) + "\n")
        # Check if we should trigger training
        if count_feedback_entries(FEEDBACK_DATASET_PATH) % FEEDBACK_TRAIN_THRESHOLD == 0:
            train_with_feedback()
    except Exception as e:
        print(f"[Feedback logging error]: {e}")
