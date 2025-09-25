import streamlit as st
from lg_backend import initialize_models, sonar_search, gpt_processing, GPT_PROMPT

st.title("KDijagnostika Automotive Diagnostic Support")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with New Chat button
with st.sidebar:
    if st.button("New Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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
        # Web search status
        with st.status("🔍 Searching web for diagnostic information...") as status:
            gpt_model, perplexity_client = initialize_models()
            sonar_response = sonar_search(user_question, perplexity_client)
            status.update(label="✅ Web search completed", state="complete")
        
        # Display sonar response
        st.subheader("🌐 Web Search Results:")
        st.write(sonar_response)
        
        # GPT processing status
        with st.status("🤖 Formatting diagnostic response...") as status:
            formatted_prompt = GPT_PROMPT.format(user_question=user_question, sonar_response=sonar_response)
            status.update(label="✅ Response ready", state="complete")
        
        # Stream GPT response
        st.subheader("🔧 Professional Diagnosis:")
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in gpt_model.stream(formatted_prompt):
            full_response += chunk.content
            response_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    complete_response = f"""🌐 **Web Search Results:**
    {sonar_response}

    🔧 **Professional Diagnosis:**
    {full_response}"""
    
    st.session_state.messages.append({"role": "assistant", "content": complete_response})