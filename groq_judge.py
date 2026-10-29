from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

load_dotenv()


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge model backed by Groq, so metrics don't need an OpenAI key."""

    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        return ChatGroq(model=self.model_name, temperature=0)

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self.model.invoke(prompt).content
        # DeepEval passes a pydantic schema and expects an instance of it back.
        # json_schema mode: gpt-oss answers in JSON rather than calling a forced tool.
        return self.model.with_structured_output(schema, method="json_schema").invoke(prompt)

    async def a_generate(self, prompt: str, schema=None):
        if schema is None:
            return (await self.model.ainvoke(prompt)).content
        return await self.model.with_structured_output(schema, method="json_schema").ainvoke(prompt)

    def get_model_name(self):
        return self.model_name
