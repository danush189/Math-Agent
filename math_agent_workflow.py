# math_agent_workflow.py
# Agentic workflow for math problem solving with knowledge base search

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import openai
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# Base Agent State Classes
@dataclass
class AgentState:
    """Base state class for all agents"""
    success: bool = True
    error: Optional[str] = None
    trace: List[str] = None

    def __post_init__(self):
        if self.trace is None:
            self.trace = []

    def add_trace(self, message: str):
        if self.trace is None:
            self.trace = []
        self.trace.append(message)

@dataclass
class MathProblemState(AgentState):
    """State for the entire math problem workflow"""
    question: str
    retrieved_info: Optional[Dict] = None
    web_search_results: Optional[Dict] = None
    solution: Optional[str] = None
    verification_result: Optional[Dict] = None
    revised_solution: Optional[str] = None
    final_verification: Optional[Dict] = None
    feedback_note: Optional[str] = None
    optimized_prompt: Optional[str] = None

# Base Agent Class
class Agent(ABC):
    """Abstract base class for all agents in the workflow"""
    def __init__(self):
        self.load_environment()
    
    def load_environment(self):
        load_dotenv()
    
    @abstractmethod
    def run(self, state: MathProblemState) -> MathProblemState:
        """Execute the agent's primary task"""
        pass

# Concrete Agent Implementations
class RetrievalAgent(Agent):
    """Agent responsible for retrieving similar problems from knowledge base"""
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = "math_problems"

    def run(self, state: MathProblemState) -> MathProblemState:
        try:
            query_embedding = self.model.encode([state.question])[0]
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=1
            )
            
            if results:
                r = results[0]
                state.retrieved_info = {
                    "question": r.payload["question"],
                    "answer": r.payload["answer"],
                    "topic": r.payload.get("topic", "unknown"),
                    "difficulty": r.payload.get("difficulty", "unknown"),
                    "score": r.score
                }
                state.add_trace(f"Retrieved similar problem with score {r.score:.2f}")
            else:
                state.add_trace("No similar problems found in knowledge base")
                
        except Exception as e:
            state.success = False
            state.error = f"Retrieval error: {str(e)}"
            state.add_trace(f"Retrieval failed: {str(e)}")
        
        return state

class WebSearchAgent(Agent):
    """Agent responsible for web search fallback"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_client = TavilyClient(self.api_key)
        self.domains = [
            "khanacademy.org", "mathstackexchange.com", "mathoverflow.net", "wolfram.com",
            "brilliant.org", "artofproblemsolving.com", "mathisfun.com", "purplemath.com",
            "symbolab.com", "desmos.com", "byjus.com", "cuemath.com", "mathhelp.com",
            "mathway.com", "chegg.com", "socratic.org", "edx.org", "coursera.org",
            "openstax.org", "mit.edu", "stanford.edu", "harvard.edu"
        ]

    def is_relevant(self, result: Dict, question: str) -> bool:
        content = result.get('content', '').lower()
        question = question.lower()
        qwords = [w for w in question.split() if len(w) > 2]
        match_count = sum(1 for w in qwords if w in content)
        return match_count >= max(1, len(qwords) // 2)

    def run(self, state: MathProblemState) -> MathProblemState:
        try:
            response = self.tavily_client.search(
                query=f"mathematical solution step by step {state.question}",
                search_depth="advanced",
                max_results=5,
                include_domains=self.domains
            )
            
            if response and response.get('results'):
                filtered = [r for r in response['results'] if self.is_relevant(r, state.question)]
                results = filtered[:1] if filtered else response['results'][:1]
                
                state.web_search_results = {
                    'content': "\n\n".join([
                        f"Source: {r.get('url', 'N/A')}\nContent: {r.get('content', 'No answer found.').strip()}"
                        for r in results
                    ])
                }
                state.add_trace(f"Found {len(results)} relevant web results")
            else:
                state.add_trace("No relevant web results found")
                
        except Exception as e:
            state.success = False
            state.error = f"Web search error: {str(e)}"
            state.add_trace(f"Web search failed: {str(e)}")
        
        return state

class SolutionAgent(Agent):
    """Agent responsible for generating math solutions using LLM"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

    def get_prompt(self, state: MathProblemState) -> str:
        if state.optimized_prompt:
            return state.optimized_prompt.format(
                question=state.question,
                retrieved_info=state.retrieved_info.get('answer') if state.retrieved_info else state.web_search_results.get('content', ''),
                feedback_note=state.feedback_note or ''
            )
        
        return f"""
You are a helpful math tutor. Given the following math question and retrieved information, provide a clear, step-by-step solution.

For each step, use a short explanation (if needed), then put the equation on its own line, wrapped in $$ ... $$. Use LaTeX for all math symbols and expressions. Do NOT use $...$ for entire sentences or explanations.

If you use square roots, write them as \\sqrt{{...}}. For exponents, use ^. For fractions, use \\frac{{a}}{{b}}.

Be concise and only show the essential steps and final answer. If the solution requires more steps, do not omit any important step.

Question: {state.question}

Retrieved Information:
{state.retrieved_info.get('answer') if state.retrieved_info else state.web_search_results.get('content', '')}

Step-by-step solution:
"""

    def run(self, state: MathProblemState) -> MathProblemState:
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key not set")
                
            prompt = self.get_prompt(state)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.2,
                top_p=0.95
            )
            
            state.solution = response.choices[0].message.content.strip()
            state.add_trace("Generated solution using LLM")
            
        except Exception as e:
            state.success = False
            state.error = f"Solution generation error: {str(e)}"
            state.add_trace(f"Solution generation failed: {str(e)}")
        
        return state

class VerificationAgent(Agent):
    """Agent responsible for verifying solutions and suggesting revisions"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

    def verify_solution(self, question: str, solution: str) -> Dict:
        # Independent solution
        prompt_independent = f"""
You are a math professor. Solve the following question step by step, as if you have not seen any previous answer. Format all math in LaTeX.

Question: {question}

Step-by-step solution:
"""
        response1 = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_independent}],
            max_tokens=512,
            temperature=0.2
        )
        independent_solution = response1.choices[0].message.content.strip()

        # Backtrack check
        prompt_backtrack = f"""
You are a math professor. Given the following answer, try to reconstruct the original question or check if the answer is a valid solution to the question. Explain your reasoning step by step.

Question: {question}

Provided Answer: {solution}

Backtrack/Validation:
"""
        response2 = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_backtrack}],
            max_tokens=512,
            temperature=0.2
        )
        backtrack_check = response2.choices[0].message.content.strip()

        # Verdict
        verdict = "Verified" if "correct" in backtrack_check.lower() or "valid" in backtrack_check.lower() else "Needs Review"
        
        return {
            "independent_solution": independent_solution,
            "backtrack_check": backtrack_check,
            "verdict": verdict
        }

    def revise_solution(self, question: str, original_solution: str, verification_feedback: str) -> str:
        prompt = f"""
You are a math professor. The following solution was flagged as needing review. Here is the original question and the previous answer. Please revise the answer to address any issues and provide a clear, correct, step-by-step solution in LaTeX.

Question: {question}

Previous Answer: {original_solution}

Verification Feedback: {verification_feedback}

Revised Step-by-step Solution:
"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def run(self, state: MathProblemState) -> MathProblemState:
        try:
            # Initial verification
            state.verification_result = self.verify_solution(state.question, state.solution)
            state.add_trace(f"Initial verification verdict: {state.verification_result['verdict']}")
            
            # If verification fails, attempt revision
            if state.verification_result['verdict'] != "Verified":
                state.revised_solution = self.revise_solution(
                    state.question,
                    state.solution,
                    state.verification_result['backtrack_check']
                )
                state.add_trace("Generated revised solution")
                
                # Verify the revised solution
                state.final_verification = self.verify_solution(state.question, state.revised_solution)
                state.add_trace(f"Final verification verdict: {state.final_verification['verdict']}")
                
        except Exception as e:
            state.success = False
            state.error = f"Verification error: {str(e)}"
            state.add_trace(f"Verification failed: {str(e)}")
        
        return state

class MathAgentWorkflow:
    """Main workflow coordinator"""
    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.web_search_agent = WebSearchAgent()
        self.solution_agent = SolutionAgent()
        self.verification_agent = VerificationAgent()

    def solve(self, question: str, feedback_note: Optional[str] = None, optimized_prompt: Optional[str] = None) -> Dict:
        # Initialize state
        state = MathProblemState(
            question=question,
            feedback_note=feedback_note,
            optimized_prompt=optimized_prompt
        )
        
        try:
            # Step 1: Knowledge base retrieval
            state = self.retrieval_agent.run(state)
            
            # Step 2: Web search if needed
            if not state.retrieved_info or state.retrieved_info.get("score", 0) < 0.8:
                state = self.web_search_agent.run(state)
            
            # Step 3: Generate solution
            state = self.solution_agent.run(state)
            
            # Step 4: Verify and possibly revise
            state = self.verification_agent.run(state)
            
            # Prepare result
            result = {
                "question": question,
                "solution": state.solution,
                "verification": state.verification_result,
                "trace": state.trace
            }
            
            if state.revised_solution:
                result.update({
                    "revised_solution": state.revised_solution,
                    "final_verification": state.final_verification
                })
            
            if not state.success:
                result["error"] = state.error
                
            return result
            
        except Exception as e:
            return {
                "error": f"Workflow error: {str(e)}",
                "trace": state.trace
            }

# For direct script execution
if __name__ == "__main__":
    workflow = MathAgentWorkflow()
    user_query = input("Enter your math question: ")
    result = workflow.solve(user_query)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        print("\nExecution trace:")
        for step in result.get('trace', []):
            print(f"- {step}")
    else:
        print("\nSolution:")
        print(result['solution'])
        print("\nVerification:")
        print(f"Verdict: {result['verification']['verdict']}")
        print(f"Backtrack Check: {result['verification']['backtrack_check']}")
        
        if 'revised_solution' in result:
            print("\nRevised Solution:")
            print(result['revised_solution'])
            print(f"\nFinal Verification: {result['final_verification']['verdict']}")
        
        print("\nExecution trace:")
        for step in result['trace']:
            print(f"- {step}")
