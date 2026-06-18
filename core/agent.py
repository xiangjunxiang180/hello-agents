from hello_agents.core.message import MessageHistory
from hello_agents.core.llm import chat
from hello_agents.tools.tool_registry import ToolRegistry


class Agent:

    def __init__(
            self,
            system_prompt=""
    ):

        self.history = MessageHistory()
        self.registry = ToolRegistry()

        if system_prompt:

            self.history.add_system(
                system_prompt
            )

    def run(self, user_input):

        self.history.add_user(
            user_input
        )

        response = chat(
            self.history.get_messages()
        )

        self.history.add_assistant(
            response
        )

        return response

    def add_tool(self, tool):

        self.registry.register(tool)
    def use_tool(
        self,
        tool_name,
        query
    ):

        tool = self.registry.get(
            tool_name
        )

        if tool is None:

            return "工具不存在"

        return tool.run(query)
