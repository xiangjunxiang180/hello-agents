from dataclasses import dataclass

@dataclass
class Message:

    role: str
    content: str


class MessageHistory:

    def __init__(self):

        self.messages = []

    def add_user(self, content):

        self.messages.append(
            {
                "role": "user",
                "content": content
            }
        )

    def add_assistant(self, content):

        self.messages.append(
            {
                "role": "assistant",
                "content": content
            }
        )

    def add_system(self, content):

        self.messages.append(
            {
                "role": "system",
                "content": content
            }
        )

    def get_messages(self):

        return self.messages
