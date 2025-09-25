from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
from langchain_openai import ChatOpenAI
from openai import OpenAI
import streamlit as st

# System prompts
SONAR_PROMPT = """
Search the web for current technical information about automotive diagnostics. 
Find detailed troubleshooting information, causes, solutions, and technical specifications.
Include sources and provide comprehensive technical data that will help diagnose automotive issues.
"""

GPT_PROMPT = """
You are KDijagnostika Support - professional automotive diagnostic technician.

Transform the web search results below into a structured diagnostic response:

**Format required:**
- **Diagnosis**: What's likely happening (1-2 lines)
- **Likely causes**: Top 3-4 causes ranked by probability
- **Quick checks**: Step-by-step troubleshooting actions
- **If still stuck**: Next recommended steps
- **Our solution**: When relevant, mention our tested Delphi/Autocom interfaces

Use professional tone, be specific with technical details, make it actionable for technicians.

---
ORIGINAL QUESTION: {user_question}
---
WEB SEARCH DATA: {sonar_response}
---
"""

# State definition
class MainState(TypedDict):
    user_question: str
    sonar_response: str
    final_answer: str

# Initialize models function
def initialize_models():
    openai_key = st.secrets["OPENAI_API_KEY"]
    perplexity_key = st.secrets["PERPLEXITY_API_KEY"]
    
    gpt_model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    perplexity_client = OpenAI(api_key=perplexity_key, base_url="https://api.perplexity.ai/")
    
    return gpt_model, perplexity_client

# Node functions
def sonar_search_node(state: MainState) -> MainState:
    _, perplexity_client = initialize_models()
    
    combined_prompt = f"{SONAR_PROMPT}\n\nUser question: {state['user_question']}"
    
    response = perplexity_client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0,
        max_tokens=1500
    )
    
    state["sonar_response"] = response.choices[0].message.content
    return state

def gpt_processing_node(state: MainState) -> MainState:
    gpt_model, _ = initialize_models()
    
    formatted_prompt = GPT_PROMPT.format(
        user_question=state["user_question"],
        sonar_response=state["sonar_response"]
    )
    
    response = gpt_model.invoke(formatted_prompt)
    state["final_answer"] = response.content
    return state

# Build LangGraph workflow
def create_workflow():
    # Initialize checkpoint
    checkpoint = InMemorySaver()
    
    # Create graph
    graph = StateGraph(MainState)
    graph.add_node("sonar_search", sonar_search_node)
    graph.add_node("gpt_processing", gpt_processing_node)
    
    # Add edges
    graph.add_edge(START, "sonar_search")
    graph.add_edge("sonar_search", "gpt_processing")
    graph.add_edge("gpt_processing", END)
    
    # Compile with checkpointer
    workflow = graph.compile(checkpointer=checkpoint)
    return workflow

# Workflow execution function
def run_diagnostic_workflow(user_question, thread_id="default"):
    workflow = create_workflow()
    
    initial_state = {
        "user_question": user_question,
        "sonar_response": "",
        "final_answer": ""
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    result = workflow.invoke(initial_state, config)
    
    return result["sonar_response"], result["final_answer"]

# Streaming workflow execution
def stream_diagnostic_workflow(user_question, thread_id="default"):
    workflow = create_workflow()
    
    initial_state = {
        "user_question": user_question,
        "sonar_response": "",
        "final_answer": ""
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    for event in workflow.stream(initial_state, config):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        yield node_name, node_output

# API endpoint function
def handle_diagnostic_request(request_data):
    user_question = request_data.get("question", "")
    thread_id = request_data.get("thread_id", "default")
    
    if not user_question.strip():
        return {"error": "No question provided"}
    
    try:
        sonar_response, final_answer = run_diagnostic_workflow(user_question, thread_id)
        return {
            "sonar_response": sonar_response,
            "final_answer": final_answer
        }
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
