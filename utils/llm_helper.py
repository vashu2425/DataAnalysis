import os
from dotenv import load_dotenv
import requests
import streamlit as st
import json

class LLMHelper:
    def __init__(self, model_name="google/flan-t5-base"):  # Changed to a smaller, faster model
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

    def generate_text(self, prompt: str, max_new_tokens=150) -> str:
        """
        Use Hugging Face Inference API to generate text remotely.
        """
        if not self.hf_token:
            st.error("Missing HF_TOKEN. Please set it in your .env file.")
            return "Error: Missing HF_TOKEN. Please set it in your .env file."

        # Format the prompt for better results
        formatted_prompt = f"""Task: Analyze the following dataset and answer the question.
Dataset Info:
{prompt}

Answer:"""

        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.3,  # Slightly increased for more creative responses
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True
            }
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # Handle different response formats
            if isinstance(result, list):
                if len(result) > 0:
                    if "generated_text" in result[0]:
                        return result[0]["generated_text"]
                    elif isinstance(result[0], str):
                        return result[0]
                return "No response generated."
            elif isinstance(result, dict):
                if "generated_text" in result:
                    return result["generated_text"]
                elif "text" in result:
                    return result["text"]
            return "Unexpected response format."

        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Hugging Face API: {str(e)}"
            st.error(error_msg)
            return error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing API response: {str(e)}"
            st.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            st.error(error_msg)
            return error_msg
