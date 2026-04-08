from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langchain.agents import create_agent
from langgraph.prebuilt import ToolNode
from react import llm, tools

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to answer question.
"""

def run_agent_reasoning(state:MessagesState) -> MessagesState:
    """Run the agent reasoning node."""

    response = llm.invoke([{
        "role": "system",
        "content" : SYSTEM_PROMPT}, *state["messages"]
    ])

    return {"messages": [response]}

tool_node = ToolNode(tools)

