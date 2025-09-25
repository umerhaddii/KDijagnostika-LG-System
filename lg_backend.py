import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI

def initialize_models():
    openai_key = st.secrets["OPENAI_API_KEY"]
    perplexity_key = st.secrets["PERPLEXITY_API_KEY"]
    gpt_model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    perplexity_sonar = OpenAI(api_key=perplexity_key, base_url="https://api.perplexity.ai/")
    return gpt_model, perplexity_sonar

SONAR_PROMPT = """Search the web for current technical information about automotive diagnostics. 
Find detailed troubleshooting information, causes, solutions, and technical specifications.
Include sources and provide comprehensive technical data that will help diagnose automotive issues."""

def sonar_search(user_question, perplexity_sonar):
    combined_prompt = f"{SONAR_PROMPT}\n\nUser question: {user_question}"
    response = perplexity_sonar.chat.completions.create(
    model="sonar",
    messages=[{"role": "user", "content": combined_prompt}],
    temperature=0,
    max_tokens=1500)
    return response.choices[0].message.content

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

def gpt_processing(user_question, sonar_response, gpt_model):
    formatted_prompt = GPT_PROMPT.format(user_question=user_question, sonar_response=sonar_response)
    response = gpt_model.invoke(formatted_prompt)
    return response.content

def run_diagnostic_workflow(user_question):
    gpt_model, perplexity_client = initialize_models()
    sonar_response = sonar_search(user_question, perplexity_client)
    final_answer = gpt_processing(user_question, sonar_response, gpt_model)
    return final_answer

def handle_diagnostic_request(request_data):
    user_question = request_data.get("question", "")
    if not user_question:
        return {"error": "No question provided."}
    try:
        result = run_diagnostic_workflow(user_question)
        return {"answer": result}
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
    

    
