import asyncio
from agents import Agent, GuardrailFunctionOutput, HandoffOutputItem, ItemHelpers, MessageOutputItem, RunContextWrapper, Runner, TResponseInputItem, ToolCallItem, ToolCallOutputItem, WebSearchTool, function_tool, input_guardrail
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel

class UserInfoContext(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    city: str | None = None

class TripPlanningGuardrailOutput(BaseModel):
    is_trip_planning: bool
    reasoning: str


@input_guardrail
async def trip_planning_guardrail( 
    ctx: RunContextWrapper[UserInfoContext], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=not result.final_output.is_trip_planning,
    )

@function_tool
async def get_city_weather(contextWrapper: RunContextWrapper[UserInfoContext], city: str) -> str:
    """Get the weather in a city.
    Args:
        city: The city to get the weather for.
    Returns:
        The weather in the city.
    """

    contextWrapper.context.city = city
    return f"The weather in {city} is sunny {contextWrapper.context.first_name}."

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="Check if the user is asking you a request that is related to trip planning.",
    output_type=TripPlanningGuardrailOutput,
)

city_info_agent = Agent[UserInfoContext](
    name="City Info Agent",
    handoff_description="A helpful agent that can answer questions about a city.",
    instructions=f"""
    {RECOMMENDED_PROMPT_PREFIX}
    You are a city info agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for the city name if not clear from the context.
    2. Use the web search tool to get information about restaurants in the city.
    3. Use the weather tool to get the live weather in the city, use your knowledge for climate questions.
    4. If the customer asks a question that is not related to the routine, transfer back to the triage agent. """,
    model="gpt-4o",
    tools=[WebSearchTool(), get_city_weather]
)

flight_finder_agent = Agent[UserInfoContext](
    name="Flight Finder Agent",
    handoff_description="A helpful agent that can find flights for a customer.",
    instructions=f"""
    {RECOMMENDED_PROMPT_PREFIX}
    You are a flight finder agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for the city name, if not clear.
    2. Use the web search tool to get information about flights to the city.
    3. If the customer asks a question that is not related to the routine, transfer back to the triage agent. """,
    model="gpt-4o-mini",
    tools=[WebSearchTool()],
)

router_agent = Agent[UserInfoContext](
    name="Router Agent",
    input_guardrails=[trip_planning_guardrail],
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}"
        "You are a helpful routing agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        city_info_agent,
        flight_finder_agent,
    ],
)

city_info_agent.handoffs.append(router_agent)
flight_finder_agent.handoffs.append(router_agent)


async def main():
    # Current agent can help reduce latency, when requests are similar. e.g, if the last question was to the city agent, current agent will = city agent, so the next question doesnt have to pass through the router. This means if the users input is about the city agent again, we have avoidedc going through the router. 
    current_agent: Agent[UserInfoContext] = router_agent
    # The LLMS context. Users Input + LLMs Output
    input_items: list[TResponseInputItem] = []
    context = UserInfoContext(first_name="James")
    while True:
        user_input = input("Enter message: ")
        input_items.append({"content": user_input, "role": "user"})
        result = await Runner.run(current_agent, input_items, context=context, workflow_name="Agents Demo")
        # print(result.final_output)

        for new_item in result.new_items:
            agent_name = new_item.agent.name
            if isinstance(new_item, MessageOutputItem):
                print(f"\033[94m{agent_name}\033[0m: {ItemHelpers.text_message_output(new_item)}")
            elif isinstance(new_item, HandoffOutputItem):
                print(f"Handed off from \033[92m{new_item.source_agent.name}\033[0m to \033[93m{new_item.target_agent.name}\033[0m")
            elif isinstance(new_item, ToolCallItem):
                print(f"\033[95m{agent_name}\033[0m: Calling a tool")
            elif isinstance(new_item, ToolCallOutputItem):
                print(f"\033[96m{agent_name}\033[0m: Tool call output: {new_item.output}")
            else:
                print(f"\033[91m{agent_name}\033[0m: Skipping item: {new_item.__class__.__name__}")

        input_items = result.to_input_list()
        current_agent = result.last_agent


if __name__ == "__main__":
    asyncio.run(main())
