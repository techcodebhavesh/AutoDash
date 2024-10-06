from ..base import RAGdashBase
from groq import Groq as GroqClient
import os

class Groq(RAGdashBase):
    def __init__(self, config=None):
        # Initialize the Groq client for Llama API
        self.client = GroqClient(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}


    def generate_sql(self, question: str, allow_llm_to_see_data=True, **kwargs) -> str:
        # Use the super generate_sql
        sql = super().generate_sql(question,  allow_llm_to_see_data=True, **kwargs)

        # Replace "\_" with "_"
        sql = sql.replace("\\_", "_")

        return sql

    def submit_prompt(self, prompt, **kwargs) -> str:
        # Use the Groq client to interact with Llama API
        print()
        print()
        print()
        print()
        print("*"*50)
        print(prompt)
        print("*"*50)
        print()
        print()
        print()
        print()
        try:
            chat_completion = self.client.chat.completions.create(
                messages=prompt,
                model="llama3-8b-8192",  # Specify the model to be used
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error occurred: {str(e)}"
