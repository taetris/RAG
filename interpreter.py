"""
Interpret semantic changes between document versions using an OpenAI model.
"""
import os
import logging
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv 

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class ChangeInterpreter:
    """Interpret changes in compliance/legal text using an LLM."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize interpreter.

        Args:
            model: OpenAI model name (default: gpt-4o-mini)
        """
        self.model = model
        self.client = OpenAI()
        logger.info(f"Using OpenAI model: {model}")

    def interpret_change(self, v1_text: str, v2_text: str, similarity: float) -> Dict[str, str]:
        """
        Classify and summarize the change between two text versions.

        Args:
            v1_text: Clause from version 1
            v2_text: Clause from version 2
            similarity: Numeric similarity score

        Returns:
            Dictionary with 'change_type' and 'summary'
        """
        if similarity > 0.9:
            return {
                "change_type": "No Significant Change",
                "summary": "Minor rewording or stylistic adjustment; no compliance impact."
            }

        categories = [
            "New Obligation",
            "Removed Obligation",
            "Clarified Scope",
            "Increased Penalty/Risk",
            "Terminology Change",
            "Structural Change",
            "No Significant Change"
        ]

        prompt = f"""
        You are a compliance analyst. Compare these two legal clauses and:
        1. Classify the type of change using one of these categories:
           {", ".join(categories)}.
        2. Briefly explain the compliance or interpretive significance of the change.

        OLD CLAUSE:
        {v1_text}

        NEW CLAUSE:
        {v2_text}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal compliance assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            message = response.choices[0].message.content.strip()

            # Try to extract the category
            change_type = self._extract_category(message, categories)

            return {"change_type": change_type, "summary": message}

        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return {"change_type": "Error", "summary": str(e)}

    def _extract_category(self, text: str, categories: list) -> str:
        """Extract change category from model response."""
        for cat in categories:
            if cat.lower() in text.lower():
                return cat
        return "Unclear"
