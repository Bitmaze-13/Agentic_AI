import os
import inspect
import json
from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable


load_dotenv()

# Initialize OpenAI Client
# This will use OPENAI_API_KEY from your .env file
client = OpenAI()

MAX_ITERATIONS = 10
MODEL = "gpt-5-mini" # Or "gpt-4o-mini"

# --- Tools ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_tier = discount_tier.lower()
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Tool Schema (Same format as Ollama/OpenAI standard)
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "The product name"}
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "discount_tier": {"type": "string"},
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

def get_tool_description(tool_dict):
    description = []
    for tool_name , tools_fucntion in tool_dict.items():
        og_func = getattr(tools_fucntion,"__wrapped__", tools_fucntion)
        signature = inspect.signature(og_func)
        docstring = inspect.getdoc(tools_fucntion) or ""
        description.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(description)

tool_descriptions =  get_tool_description(tools_dict)
tools_name = ", ".join(tools_dict.keys())

print(f"tool description : {tool_descriptions}")
print(f"tool name: {tools_name}")

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# --- Helper: Traced OpenAI call ---
@traceable(name="OpenAI Chat", run_type="llm")
def openai_chat_traced(messages, **options):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools_for_llm,
        **options
        # tool_choice="auto"
    )

@traceable(name="OpenAI Agent Loop")
def run_agent(question: str):
    messages = [
        {
            "role": "user", 
            "content": react_prompt.format(question=question)
        }
    ]
    scratchpad = ""

    

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        
        response = openai_chat_traced(
            messages=messages
        )

        ai_message = response.choices[0].message
        
        # Add the AI's message to history (essential for OpenAI tool flow)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # Process the tool call
        tool_call = ai_message.tool_calls[0]
        tool_name = tool_call.function.name
        # OpenAI returns arguments as a JSON string, must parse them
        tool_args = json.loads(tool_call.function.arguments)

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_func = tools_dict.get(tool_name)
        observation = tool_func(**tool_args)

        print(f"  [Tool Result] {observation}")

        # IMPORTANT: OpenAI requires the tool_call_id to match the result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id, 
            "name": tool_name,
            "content": str(observation),
        })

    return None

if __name__ == "__main__":
    run_agent("What is the price of a laptop after applying a gold discount?")
