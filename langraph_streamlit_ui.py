import streamlit as st
import uuid
from langgraph_backend import initialize_models, stream_diagnostic_workflow, GPT_PROMPT

st.title("KDijagnostika Automotive Diagnostic Support")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar with New Chat button
with st.sidebar:
    if st.button("New Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if "sonar_response" in message:
                st.subheader("🌐 Web Search Results:")
                st.write(message["sonar_response"])
            if "final_answer" in message:
                st.subheader("🔧 Professional Diagnosis:")
                st.markdown(message["final_answer"])
        else:
            st.markdown(message["content"])

# Chat input
user_question = st.chat_input("Enter your automotive diagnostic question:")

if user_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Assistant response
    with st.chat_message("assistant"):
        sonar_response = ""
        final_answer = ""
        
        # Initialize status container
        status_container = st.status("🔍 Searching web for diagnostic information...")
        
        # Stream through LangGraph workflow
        for node_name, node_output in stream_diagnostic_workflow(user_question, st.session_state.thread_id):
            if node_name == "sonar_search":
                sonar_response = node_output["sonar_response"]
                status_container.update(label="✅ Web search completed", state="complete")
                
                st.subheader("🌐 Web Search Results:")
                st.write(sonar_response)
                
                # Update status for next phase
                status_container = st.status("🤖 Formatting diagnostic response...")
            
            elif node_name == "gpt_processing":
                status_container.update(label="✅ Response ready", state="complete")
                
                st.subheader("🔧 Professional Diagnosis:")
                
                # Real streaming using GPT model directly
                gpt_model, _ = initialize_models()
                formatted_prompt = GPT_PROMPT.format(
                    user_question=user_question, 
                    sonar_response=sonar_response
                )
                
                # Stream the response using st.write_stream
                final_answer = st.write_stream(
                    chunk.content for chunk in gpt_model.stream(formatted_prompt)
                )
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "sonar_response": sonar_response,
        "final_answer": final_answer
    })