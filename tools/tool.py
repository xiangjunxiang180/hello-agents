class Tool:

    def __init__(
        self,
        name,
        description
    ):
        self.name = name
        self.description = description

    def run(self, query):
        raise NotImplementedError

    def get_schema(self):
        raise NotImplementedError
