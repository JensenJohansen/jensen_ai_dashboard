from datetime import datetime, timedelta
import json
import os
import re
import pandas as pd
import requests
from vanna.base import VannaBase
from vanna.exceptions import DependencyError
from dotenv import load_dotenv

load_dotenv()  # Automatically loads from .env

api_key = os.getenv("GROK_KEY")

# Helper function to serialize non-JSON-serializable objects
def serialize_data(obj):
    if isinstance(obj, timedelta):
        return str(obj)  # Convert timedelta to string (e.g., "1 day, 2:30:00")
    elif isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO string (e.g., "2025-06-22T17:26:00")
    elif isinstance(obj, (list, tuple)):
        return [serialize_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_data(value) for key, value in obj.items()}
    elif pd.isna(obj):
        return None
    return obj

class XAI_Grok(VannaBase):
    def __init__(self, config=None):
        super().__init__(config)
        try:
            import requests
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install requests"
            )

        if not config:
            raise ValueError("config must contain at least model")
        if 'model' not in config.keys():
            raise ValueError("config must contain at least model")

        self.model = config.get('model', 'grok-3-latest')
        self.api_key = config.get('api_key',api_key) or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY must be set in config or environment")
        self.system_prompt = config.get('system_prompt', "You are Grok, a helpful AI that generates accurate SQL queries based on natural language questions.")
        self.timeout = config.get('timeout', 60.0)  # Timeout in seconds

    def system_message(self, message: str = None) -> dict:
        """Return a system message dictionary for xAI API."""
        return {
            "role": "system",
            "content": message or self.system_prompt
        }

    def user_message(self, message: str) -> dict:
        """Return a user message dictionary for xAI API."""
        return {
            "role": "user",
            "content": message
        }

    def assistant_message(self, message: str) -> dict:
        """Return an assistant message dictionary for xAI API."""
        return {
            "role": "assistant",
            "content": message
        }

    def extract_sql(self, llm_response: str) -> str:
        """
        Extracts the first SQL statement from Grok's response, handling markdown or plain text.
        Matches after 'SELECT' or 'WITH' (case-insensitive) until a semicolon, square bracket, or code block end.

        Args:
            llm_response (str): The response from Grok containing the SQL query.

        Returns:
            str: The extracted SQL statement, with markdown removed, or the original response if no SQL is found.
        """
        if not llm_response:
            self.log("No response from Grok to extract SQL")
            return ""

        # Remove any escaped characters
        llm_response = llm_response.replace("\\_", "_").replace("\\", "")

        # Regular expression to find ```sql block and capture until closing ```
        sql_block = re.search(r"```sql\n((.|\n)*?)(?=;|\[|```)", llm_response, re.DOTALL)
        # Regular expression to find SELECT or WITH (case-insensitive) and capture until ;, [, or end
        select_with = re.search(r'(select|(with.*?as \())(.*?)(?=;|\[|```|$)',
                                llm_response,
                                re.IGNORECASE | re.DOTALL)
        
        if sql_block:
            extracted_sql = sql_block.group(1).replace("```", "").strip()
            self.log(f"Output from Grok: {llm_response}\nExtracted SQL: {extracted_sql}")
            return extracted_sql
        elif select_with:
            extracted_sql = select_with.group(0).strip()
            self.log(f"Output from Grok: {llm_response}\nExtracted SQL: {extracted_sql}")
            return extracted_sql
        else:
            self.log(f"Output from Grok: {llm_response}\nNo SQL found, returning as is")
            return llm_response.strip()

    def submit_prompt(self, prompt, **kwargs) -> str:
        """Submit a prompt to xAI Grok API."""
        if not prompt:
            self.log("Empty prompt provided to xAI API")
            return ""

        try:
            messages = prompt if isinstance(prompt, list) else [
                self.system_message(),
                self.user_message(prompt)
            ]
            self.log(f"xAI API request: {json.dumps(messages, ensure_ascii=False)}")
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "messages": messages,
                    "model": self.model,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content'].strip()
            self.log(f"xAI API response: {content}")
            return content if content else ""
        except requests.exceptions.HTTPError as e:
            self.log(f"xAI API HTTP error: {e.response.status_code} - {e.response.text}")
            return ""
        except Exception as e:
            self.log(f"Error submitting prompt to xAI: {str(e)}")
            return ""