import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schema import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You're an expert researcher.
            Current Time : {time}
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.  
    """
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(
    time = lambda : datetime.datetime.now().isoformat(),
    )

first_responder_prompt = actor_prompt.partial(
    first_instruction = "Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """
Revise your previous answer using the new information.
 - You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (Which does not count towards the word limit), in the form of:
        - [1] https://example.com
        - [2] https://example.com
    - You should use the previous critiques to remove superflous information from your answer and make SURE it is not more than 250 words.
"""

reviser = actor_prompt.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")


# if __name__ =="__main__":
#     human_message = HumanMessage(
#         content="Write about AI-powered SOC / autonomous soc problem domain."
#         "list startups that do that and raised capital."
#     )

#     chain = (
#         first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
#         | parser_pydantic
#     )

#     res = chain.invoke(input={"messages": [human_message]})
#     print(res)