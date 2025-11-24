# import os
from typing import Annotated, Literal, TypedDict

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from tools.find_weather import get_weather
from tools.rag import get_retriever

QDRANT_PATH = "./qdrant_data"

@tool
def lookup_policy(query: str):
    """
    Use this tool to find information about AI Agents, prompting techniques 
    (like one-shot, few-shot, CoT), and internal AI documentation.
    Always use this tool if the user asks for an explanation of an AI concept.
    """
    client = QdrantClient(path=QDRANT_PATH) 
    
    try:
        retriever = get_retriever(client)
        docs = retriever.invoke(query)
        content = "\n\n".join([doc.page_content for doc in docs])
    finally:
        client.close()
        
    return content

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

tools = [get_weather, lookup_policy]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", "agent")

graph = workflow.compile()
