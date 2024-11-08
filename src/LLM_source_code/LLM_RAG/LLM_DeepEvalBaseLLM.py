from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel
import json

class AzureOpenAIClass(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        chat_model = self.load_model()
        print("prompt - ",prompt)
        output = chat_model.invoke(prompt)
        print("Raw output - ",output)
        print("Raw output ended$$$$$$")
        output = output.replace("\n","")
        if "<|im_end|>" in output:
            output = output.replace("<|im_end|>","")
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # chat_model = self.load_model()
        # res = await chat_model.ainvoke(prompt)
        # return res.content
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

