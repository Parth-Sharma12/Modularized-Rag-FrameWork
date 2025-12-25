from generation.gemini_client import GeminiClient

class AnswerGenerator:
    def __init__(self):
        self.llm = GeminiClient()

    def generate(self, query: str, context: str) -> str:
        prompt = f"""
            You are an assistant.
            Answer ONLY using the context below.
            If the answer is not present, say "I don't know".

            Context:
            {context}

            Question:
            {query}
        """
        print("Prompt final %s",prompt)
        return self.llm.generate(prompt)
