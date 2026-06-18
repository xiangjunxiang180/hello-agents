from hello_agents.tools.tool import Tool


class CalculatorTool(Tool):

    def __init__(self):

        super().__init__(
            name="calculator",
            description="执行数学计算"
        )

    def run(self, query):

        try:

            result = eval(query)

            return str(result)

        except Exception as e:

            return f"计算错误: {e}"
    def get_schema(self):

        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
