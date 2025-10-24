import asyncio
import operator
import os
from typing import TypedDict, Annotated, List, Literal

import httpx
from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import interrupt, Command

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    def __init__(
            self,
            llm_model_client: AzureChatOpenAI,
            memory_checkpointer: MemorySaver,
            mcp_client: Client,
            mcp_tool_list: list,
    ):
        self._llm = llm_model_client.bind_tools(mcp_tool_list)
        self._checkpointer = memory_checkpointer
        self._mcp_client = mcp_client
        self._graph = self._build_graph()
        self._system_prompt = """You are a helpful assistant that uses tools to answer questions.
        When you have gathered all information needed, provide the final answer.

        IMPORTANT RULES:
        - You can ONLY communicate with the user through the 'human_ask' tool
        - Never respond directly to the user - always use human_ask tool if you need more information
        - Use ONLY one Action at a time
        - Continue until you can provide a Final Answer
        - DO NOT finish work until you do not have full date to answer user question.
        - DO NOT use own knowledge, if you need some information either use tool 'facts_from_wikipedia' or 'human_ask'. Prioritize wikipedia"""

    def _build_graph(self) -> CompiledStateGraph:
        graph_skeleton = StateGraph(AgentState)
        graph_skeleton.add_node("call_llm_node", self._call_llm_node)
        graph_skeleton.add_node("call_human_node", self._call_human_node)
        graph_skeleton.add_node("call_mcp_node", self._call_mcp_node)
        graph_skeleton.set_entry_point("call_llm_node")

        graph_skeleton.add_conditional_edges(
            "call_llm_node",
            self._loop_route,
            {
                "call_human_node": "call_human_node",
                "call_mcp_node": "call_mcp_node",
                "end_loop": END,
            }
        )
        graph_skeleton.add_edge("call_mcp_node", "call_llm_node")
        graph_skeleton.add_edge("call_human_node", "call_llm_node")
        return graph_skeleton.compile(checkpointer=self._checkpointer)

    async def __call__(self, user_input: str, thread_id: str | None = None) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )
        current_state = self._graph.get_state(config)

        while current_state.next:
            human_input = input(f"{current_state.interrupts[0].value}")
            result = await self._graph.ainvoke(Command(resume=human_input), config)
            current_state = self._graph.get_state(config)
        return result["messages"][-1].content

    async def _call_llm_node(self, state: AgentState) -> dict:
        response = await self._llm.ainvoke(
            [SystemMessage(content=self._system_prompt)] + state["messages"]
        )

        return {
            "messages": [AIMessage(content=response.content, tool_calls=response.tool_calls)],
        }

    def _call_human_node(self, state: AgentState) -> dict:
        action_input = state['messages'][-1].tool_calls[0].get('args')
        human_response = interrupt(f"{action_input.get('question')}\nQ: ")
        observation = f"Observation: Human responded: {human_response}"
        return {
            "messages": [ToolMessage(tool_call_id=state['messages'][-1].tool_calls[0]['id'],
                                     name=state['messages'][-1].tool_calls[0]['name'], content=observation)],
        }

    async def _call_mcp_node(self, state: AgentState) -> dict:
        observation_msg = []
        all_tool_names = ','.join(t['name'] for t in state['messages'][-1].tool_calls)
        human_response = interrupt(f"Can I call tool which name is: {all_tool_names}\nQ: ")
        accept_tools = bool(human_response.lower() in ["true", "yes", "tak", "sure", "y", "1", "t"])

        for tool_call in state['messages'][-1].tool_calls:
            if accept_tools:
                r = await self._mcp_client.call_tool(tool_call.get('name'), tool_call.get('args'))
                observation_msg.append(
                    ToolMessage(
                        content=r.data,
                        tool_call_id=tool_call.get('id'),
                        name=tool_call.get('name')),
                )
            else:
                observation_msg.append(
                    ToolMessage(
                        content=f"Observation: action not allowed",
                        tool_call_id=tool_call.get('id'),
                        name=tool_call.get('name')),
                )
        return {"messages": observation_msg}

    def _loop_route(self, state: AgentState) -> Literal["call_human_node", "call_mcp_node", "end_loop"]:
        tool_calls = state['messages'][-1].tool_calls
        if tool_calls and tool_calls[0].get('name') == 'human_ask':
            return "call_human_node"
        elif tool_calls:
            return "call_mcp_node"
        else:
            return "end_loop"


async def main():
    async with Client("02_agent_mcp_server.py") as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": getattr(tool, "inputSchema", {"type": "object", "properties": {}}),
            }
            for tool in mcp_tools
        ]

        # Dodatkowe narzędzie human_ask
        tools.append({
            "name": "human_ask",
            "description": "if you need more information from user use this tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "here pass the user question"}
                },
            },
        })

        agent_ai = Agent(
            llm_model_client=AzureChatOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_ENDPOINT"],
                api_version="2024-08-01-preview",
                deployment_name=os.environ['AZURE_MODEL_NAME'],
                temperature=0,
                model=os.environ['AZURE_MODEL_NAME'],
                http_client=httpx.Client(verify=False),
                http_async_client=httpx.AsyncClient(verify=False)
            ),
            memory_checkpointer=MemorySaver(),
            mcp_client=mcp_client,
            mcp_tool_list=tools,
        )
        result = await agent_ai(
            user_input="Weather in the city you are asking the user about",
            thread_id="1",
        )
        print(result)


if __name__ == '__main__':
    asyncio.run(main())
