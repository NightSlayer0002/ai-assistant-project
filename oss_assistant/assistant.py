"""Core logic for communicating with the Qwen2.5 model via HuggingFace Inference API."""

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 512
MAX_HISTORY_TURNS = 10
SYSTEM_PROMPT = "You are a helpful personal assistant. Be concise and accurate."


class OSSAssistant:
    """Manages conversation with the Qwen2.5 model via HuggingFace Inference API."""

    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY not found. "
                "Create a .env file with your key from https://huggingface.co/settings/tokens"
            )

        self.client = InferenceClient(api_key=self.api_key)
        self.conversation_history: list[dict] = []

    def send_message(self, user_input: str) -> str:
        """Sends a user message to the model and returns the response."""
        self.conversation_history.append({"role": "user", "content": user_input})

        # Cap history to prevent token overflow
        if len(self.conversation_history) > MAX_HISTORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-(MAX_HISTORY_TURNS * 2):]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history

        try:
            response = self.client.chat_completion(
                model=MODEL_ID,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
            )

            assistant_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                return "⏳ Rate limited by HuggingFace API. Please wait a moment and try again."
            elif "503" in error_msg:
                return "🔄 Model is loading on HuggingFace servers. Please try again in ~30 seconds."
            return f"❌ API error: {error_msg}"

    def clear_history(self):
        """Resets conversation memory."""
        self.conversation_history = []

    def get_history(self) -> list[dict]:
        """Returns a copy of the conversation history."""
        return self.conversation_history.copy()
