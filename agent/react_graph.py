# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import uuid
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from toolbox_langchain import ToolboxTool

from .tools import (
    get_auth_tools,
    get_confirmation_needing_tools,
)


class UserState(TypedDict):
    """
    State with messages for each session/user.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


def __is_logged_in(config: RunnableConfig) -> bool:
    """
    Checks if the user is logged in based on the provided config.
    """
    return bool(
        config
        and "configurable" in config
        and "auth_token_getters" in config["configurable"]
        and "my_google_service" in config["configurable"]["auth_token_getters"]
        and config["configurable"]["auth_token_getters"]["my_google_service"]()
    )


def __get_tool_to_run(tool: ToolboxTool, config: RunnableConfig):
    if (
        config
        and "configurable" in config
        and "auth_token_getters" in config["configurable"]
    ):
        auth_token_getters = config["configurable"]["auth_token_getters"]
        if auth_token_getters:
            core_tool = tool._ToolboxTool__core_tool  # type:ignore
            required_auth_keys = set(core_tool._required_authz_tokens)
            for auth_list in core_tool._required_authn_params.values():
                required_auth_keys.update(auth_list)
            filtered_getters = {
                k: v for k, v in auth_token_getters.items() if k in required_auth_keys
            }
            if filtered_getters:
                return tool.add_auth_token_getters(filtered_getters)
    return tool


async def create_graph(
    tools: list[ToolboxTool],
    checkpointer: MemorySaver,
    prompt: ChatPromptTemplate,
    model_name: str,
    debug: bool,
):
    """
    Creates a simple graph for chat interactions with tool calling.
    Removed ticket booking logic to fit BCEL product inquiry use case.
    """

    # --- NODE: Execute Tools ---
    async def tool_node(state: UserState, config: RunnableConfig):
        last_message = state["messages"][-1]
        tool_messages = []

        if not hasattr(last_message, "tool_calls"):
            return {"messages": []}

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            selected_tool = next((t for t in tools if t.name == tool_name), None)

            if not selected_tool:
                output = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    tool_to_run: ToolboxTool = __get_tool_to_run(selected_tool, config)
                    output = await tool_to_run.ainvoke(tool_call["args"])
                except Exception as e:
                    output = f"Error executing tool {tool_name}: {e}"

            tool_messages.append(
                ToolMessage(
                    name=tool_to_run.name,
                    content=output,
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_messages}

    # --- NODE: Agent (LLM) ---
    model = ChatVertexAI(max_output_tokens=512, model_name=model_name, temperature=0.0)
    model_with_tools = model.bind_tools(tools)
    model_runnable = prompt | model_with_tools

    async def acall_model(state: UserState, config: RunnableConfig):
        messages = state["messages"]
        res = await model_runnable.ainvoke({"messages": messages}, config)
        return {"messages": [res]}

    # --- NODE: Request Login ---
    def request_login_node(_: UserState):
        return {
            "messages": [
                AIMessage(
                    content="This action requires you to be signed in. Please log in and then try again."
                )
            ]
        }

    # --- EDGE LOGIC ---
    def agent_should_continue(
        state: UserState, config: RunnableConfig
    ) -> Literal["continue", "request_login", "end"]:
        """
        Determine flow:
        - If no tool calls -> END
        - If tool requires auth & not logged in -> REQUEST_LOGIN
        - Otherwise -> CONTINUE (to tool node)
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or len(last_message.tool_calls) == 0:
            return "end"

        for tool_call in last_message.tool_calls:
            # Check for Auth Tools (ถ้าอนาคตมี tool ที่ต้อง login)
            if tool_call["name"] in get_auth_tools():
                if not __is_logged_in(config):
                    return "request_login"
            
            # ตัด logic booking validation ออกไปเลย
            # if tool_call["name"] in get_confirmation_needing_tools(): ...

        return "continue"

    # --- Define Graph Structure ---
    AGENT_NODE = "agent"
    TOOL_NODE = "tools"
    REQUEST_LOGIN_NODE = "request_login"

    llm_graph = StateGraph(UserState)
    llm_graph.add_node(AGENT_NODE, RunnableLambda(acall_model))
    llm_graph.add_node(TOOL_NODE, tool_node)
    llm_graph.add_node(REQUEST_LOGIN_NODE, request_login_node)

    llm_graph.set_entry_point(AGENT_NODE)

    # Edges
    llm_graph.add_conditional_edges(
        AGENT_NODE,
        agent_should_continue,
        {
            "continue": TOOL_NODE,
            "request_login": REQUEST_LOGIN_NODE,
            "end": END,
        },
    )
    llm_graph.add_edge(TOOL_NODE, AGENT_NODE)
    llm_graph.add_edge(REQUEST_LOGIN_NODE, END)

    # Compile
    langgraph_app = llm_graph.compile(checkpointer=checkpointer, debug=debug)
    return langgraph_app