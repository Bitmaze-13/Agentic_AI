from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
MAX_ITERATIONS = 10

MODEL = "gpt-5-mini"

@tool
def get_product_price(product:str) -> str:
    """ Looking the price of a product in the catalog """
    print(f">> Executing get_product_price(product='{product}')")
    prices = {"laptop":1299.99, "headphone":139.96, "Keyboard":89.50}
    return prices.get(product,0)

@tool
def get_discount(price:float, discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price. """
    print(f">> Executing the get_discount(price={price},discount_tier='{discount_tier}')")

    discount_per = {"bronze":5, "silver":12, "gold":23}

    discount_val = discount_per.get(discount_tier,0)

    return round(price * (1 - discount_val / 100), 2)


# ---------------- Agent Loop ---------------

def run_agent(question:str):
    tools = [get_product_price, get_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(model="nvidia/nemotron-3-super-120b-a12b",
                        model_provider="nvidia",
                        temperature= 0)
    
    llm_with_tools = llm.bind_tools(tools)

    print(f"Questions: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly.\n"
                "1. NEVER guess or assume any price."
                "you MUST call get_product_price first to get the price of product.\n"
                "2. ONLY apply get_discount AFTER you have the price"
                "received from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made up number. \n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]

    for iterations in range(1, MAX_ITERATIONS + 1):
        print(f"\n iterations: {iterations}")
        
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n final answer: {ai_message.content}.")
            return ai_message.content
        
        # process only the FIRST tool call - force one tool per iteration

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        print(f"Tool Selected: {tool_name} with args : {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError({f"tool: {tool_name} not found"})
        
        observations = tool_to_use.invoke(tool_args)
        
        print(f"tools result : {observations}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observations), tool_call_id= tool_call_id
            ))

    print(f"Max iterations REACHED without Final Answer ")
    return None
     




if __name__ == "__main__":
    print("Hello This is Neel's Agent")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")

