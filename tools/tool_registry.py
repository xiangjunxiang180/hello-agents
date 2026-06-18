class ToolRegistry:

    def __init__(self):

        self.tools = {}

    def register(self, tool):

        self.tools[tool.name] = tool

    def get(self, name):

        return self.tools.get(name)

    def list_tools(self):

        return list(self.tools.keys())
    def get_schemas(self):

        return [
            tool.get_schema()
            for tool in self.tools.values()
        ]
