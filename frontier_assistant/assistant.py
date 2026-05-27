"""Core conversation logic and Gemini API integration with tool use."""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from frontier_assistant.tools import TOOL_DEFINITIONS, execute_tool

load_dotenv()

MODEL_ID = "gemini-2.5-flash"
MAX_TOKENS = 1024
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = (
    "You are a helpful personal assistant. Be concise and accurate. "
    "You have access to tools for getting the current time and performing calculations."
)


class FrontierAssistant:
    """Manages conversation with Gemini, including tool use."""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Create a .env file with your key from https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.conversation_history = []

    def send_message(self, user_input: str) -> str:
        """Sends a message to Gemini and handles the tool use loop if needed."""
        self.conversation_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        )

        if len(self.conversation_history) > MAX_HISTORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-(MAX_HISTORY_TURNS * 2):]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_TOKENS,
            temperature=0.7,
            tools=[TOOL_DEFINITIONS],
        )

        try:
            response = self.client.models.generate_content(
                model=MODEL_ID,
                contents=self.conversation_history,
                config=config,
            )

            # Tool use loop: execute function calls and feed results back to Gemini
            while response.candidates and response.candidates[0].content.parts and any(
                hasattr(part, 'function_call') and part.function_call
                for part in response.candidates[0].content.parts
            ):
                self.conversation_history.append(response.candidates[0].content)

                function_response_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        result = execute_tool(
                            part.function_call.name,
                            dict(part.function_call.args) if part.function_call.args else {}
                        )
                        function_response_parts.append(
                            types.Part.from_function_response(
                                name=part.function_call.name,
                                response={"result": result},
                            )
                        )

                # Function results are sent back as user-role content (Gemini API convention)
                self.conversation_history.append(
                    types.Content(role="user", parts=function_response_parts)
                )

                response = self.client.models.generate_content(
                    model=MODEL_ID,
                    contents=self.conversation_history,
                    config=config,
                )

            assistant_message = response.text or ""
            self.conversation_history.append(response.candidates[0].content)
            return assistant_message

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                import time as _time
                if self.conversation_history and self.conversation_history[-1].role == "user":
                    self.conversation_history.pop()
                print(f"    ⏳ Rate limited. Waiting 15s before retry...")
                _time.sleep(15)
                try:
                    return self.send_message(user_input)
                except Exception:
                    return "⏳ Rate limited. Please wait a few minutes and try again."
            return f"❌ API error: {error_msg}"

    def clear_history(self):
        """Resets conversation memory."""
        self.conversation_history = []

    def get_history(self):
        """Returns a copy of conversation history."""
        return self.conversation_history.copy()
