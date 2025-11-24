import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph_agent import graph

st.set_page_config(page_title="AI Assistant and weather reporter", layout="centered")

st.title("AI Agent: Weather & AI Agent Expert")
st.caption("I can answer questions about the weather or AI Agents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        inputs = {"messages": st.session_state.messages}
        result = graph.invoke(inputs)
        final_response = result["messages"][-1]
        content = final_response.content
        # print(final_response.content)
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            response_text = " ".join(text_parts)
        else:
            response_text = str(content)
        
        message_placeholder.markdown(response_text)
    
    st.session_state.messages.append(AIMessage(content=response_text))
