import json
from typing import Any, AsyncGenerator, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from esdc.chat.prompts import get_system_prompt
from esdc.chat.tools import execute_sql, get_schema, list_tables


def create_agent(
    llm: BaseChatModel,
    tools: list | None = None,
) -> Any:
    """Create a LangGraph agent with tools.

    Args:
        llm: A LangChain chat model
        tools: Optional list of tools. Defaults to [execute_sql, get_schema, list_tables]

    Returns:
        A compiled StateGraph agent
    """
    if tools is None:
        tools = [execute_sql, get_schema, list_tables]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Agent node that calls the LLM with tools."""
        system_prompt = get_system_prompt()
        messages_with_system = [SystemMessage(content=system_prompt)] + state[
            "messages"
        ]

        response = llm_with_tools.invoke(messages_with_system)
        return {"messages": [cast(AnyMessage, response)]}

    def should_continue(state: MessagesState) -> str:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        ai_message = cast(AIMessage, last_message)
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            return "tools"

        return END

    def tool_node(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Tool execution node."""
        result = []
        last_message = state["messages"][-1]

        ai_message = cast(AIMessage, last_message)
        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            return {"messages": []}

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                try:
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                    observation = tool.invoke(tool_args)
                except Exception as e:
                    observation = f"Error: {str(e)}"

                result.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": tool_name,
                        "content": observation,
                    }
                )

        return {"messages": result}

    graph = (
        StateGraph(MessagesState)
        .add_node("agent", agent_node)
        .add_node("tools", tool_node)
        .add_edge(START, "agent")
        .add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            },
        )
        .add_edge("tools", "agent")
    )

    return graph.compile()


async def run_agent_stream(
    agent: Any,
    user_input: str,
    thread_id: str,
    checkpointer: Any | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the agent with streaming output.

    Args:
        agent: Compiled LangGraph agent
        user_input: User message
        thread_id: Conversation thread ID
        checkpointer: Optional checkpointer for memory

    Yields:
        Dict with 'type' (message/tool/error) and 'content'
    """
    config = {"configurable": {"thread_id": thread_id}}

    messages = [HumanMessage(content=user_input)]

    if checkpointer:
        agent = agent.compile(checkpointer=checkpointer)
        config["configurable"]["checkpoint_ns"] = "esdc_chat"

    stored_tool_call: dict[str, Any] = {}

    async for chunk in agent.astream(
        {"messages": messages},
        config=config,
        stream_mode="values",
    ):
        if "messages" in chunk:
            last_msg = chunk["messages"][-1]

            # Check for ToolMessage first (tool result - has name, no tool_calls)
            if (
                hasattr(last_msg, "name")
                and last_msg.name
                and not hasattr(last_msg, "tool_calls")
            ):
                tool_name = last_msg.name
                tool_result = last_msg.content if hasattr(last_msg, "content") else ""
                sql = ""
                if tool_name == "execute_sql" and stored_tool_call.get("args"):
                    args = stored_tool_call.get("args", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    sql = args.get("query", "")
                yield {
                    "type": "tool_result",
                    "tool": tool_name,
                    "result": tool_result,
                    "sql": sql,
                }
            # Only yield AIMessage content (not HumanMessage or other types)
            elif isinstance(last_msg, AIMessage):
                if last_msg.content:
                    yield {
                        "type": "message",
                        "content": last_msg.content,
                        "additional_kwargs": last_msg.additional_kwargs
                        if hasattr(last_msg, "additional_kwargs")
                        else {},
                    }
                elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        stored_tool_call = {
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                        }
                        yield {
                            "type": "tool_call",
                            "tool": tc["name"],
                            "args": tc.get("args", {}),
                        }
