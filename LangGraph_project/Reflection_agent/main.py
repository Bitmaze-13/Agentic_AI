from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from chains import generation_chain, reflection_chain
load_dotenv(dotenv_path="D:\GenAI_Langchain\Agentic_AI\LangGraph_project\.env")

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages":state["messages"]})]}

def reflection_node(state:MessageGraph):
    res = reflection_chain.invoke({"messages":state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

def should_continue(state:MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

reflection_flow = StateGraph(state_schema=MessageGraph)

reflection_flow.add_node(GENERATE, generation_node)
reflection_flow.add_node(REFLECT, reflection_node)
reflection_flow.set_entry_point(GENERATE)


reflection_flow.add_conditional_edges(GENERATE, should_continue, path_map={END:END, REFLECT:REFLECT})
reflection_flow.add_edge(REFLECT, GENERATE)

graph = reflection_flow.compile()
print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    inputs = HumanMessage(
        content="""Make this Better
        @WorldOfAI
        AI isn't going to replace you, but a person using AI might. 
        The future belongs to those who treat these models as their co-pilot rather than a threat. 
        Adapt or get left behind. 🤖🚀 #AI #FutureOfWork"""
    )

    res = graph.invoke(inputs)
    print(res)