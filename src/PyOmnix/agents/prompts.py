# prompts.py
import json
from dataclasses import dataclass, field
from pathlib import Path

from pyomnix.consts import OMNIX_PATH
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


@dataclass
class PromptRegistry:
    """
    A registry of prompts for the agent.
    """

    default: str = (
        "You are a helpful, knowledgeable, and friendly AI assistant.\n"
        "You provide clear, accurate, and thoughtful responses "
        "while being respectful and professional.\n"
        "[Language Rules]\n"
        "- Respond in the same language the user uses."
    )
    summary: str = (
        "You are a professional conversation summarizer.\n"
        "Your task is to create a concise summary of the conversation.\n"
        "[Instructions]\n"
        "1. Capture the key topics(highlight the overall topic at the beginning), decisions, and important context.\n"
        "2. Preserve the core meaning, key details, tone and style, participant roles and dialog structure.\n"
        "3. Keep the summary concise but comprehensive (usually 2-5 sentences).\n"
        "4. Focus on information useful for continuing the conversation later.\n"
        "[Language Rules]\n"
        "- CRITICAL: Maintain the original language of the conversation."
    )
    summary_update: str = (
        "You are a professional conversation summarizer.\n"
        "Your task is to update an existing summary with new conversation content.\n"
        "[Instructions]\n"
        "1. Merge the existing summary with the new conversation content.\n"
        "2. Keep the updated summary concise but comprehensive.\n"
        "3. Remove outdated information if it's been superseded.\n"
        "[Language Rules]\n"
        "- CRITICAL: Maintain the original language of the conversation.\n"
        "- Match the language of the existing summary."
    )
    query_rewrite: str = (
        "You are a query rewriting specialist.\n"
        "Your task is to rephrase a follow-up question into a standalone query.\n"
        "[Instructions]\n"
        '1. Resolve all pronouns and references (e.g., "it", "that", "they").\n'
        "2. Include necessary context from the conversation in the rephrased query.\n"
        "3. Keep the query concise and search-friendly.\n"
        "4. If the question is already standalone, return it as-is.\n"
        "[Language Rules]\n"
        "- CRITICAL: Maintain the original language of the question."
    )
    critic: str = (
        "### ROLE: CRITICAL AUDITOR\n"
        "Your mission is to rigorously audit the 'Debater's' output.\n" "Ignore the user query's direct answer; focus exclusively on identifying systemic flaws.\n"
        "Apply the principle: **Truth emerges through friction.**\n"
        
        "### EXECUTION FILTERS\n"
        "1. **Logic Check**: Identify circular reasoning, non-sequiturs, or 'hand-waving' (vague abstractions of complex mechanisms).\n"
        "2. **Empirical Audit**: Flag hallucinations, speculative assertions disguised as facts, and missing citations/evidence.\n"
        "3. **Boundary Analysis**: Stress-test the argument against edge cases and overlooked counter-perspectives.\n"
        "4. **Bias Detection**: Expose over-generalizations or 'alignment bias' (sacrificing truth for agreeableness).\n"
        "### OUTPUT STRUCTURE\n"
        "**1. CRITICAL ANALYSIS**\n"
        "[Specific errors/fallacies/gaps]\n"
        "**2. COUNTER-POINTS**\n"
        "[The strongest opposing argument or failure state]\n"
        "**3. REVISION DIRECTIVES**\n"
        "[Hard constraints/instructions for the next iteration]\n"
    )
    extra_prompts: dict = field(default_factory=dict)

    def get(self, key: str) -> str:
        """
        Get a prompt by key.
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra_prompts.get(key, f"Prompt {key} not found")

    def load_overrides(self, path: Path | str | None = None):
        """
        Load overrides from a prompts.json file.
        If path is not provided, use OMNIX_PATH / "prompts.json".
        If OMNIX_PATH is not set, return.
        If path is not a valid file, return.
        """
        if path is None:
            if OMNIX_PATH is not None:
                path = OMNIX_PATH / "prompts.json"
            else:
                logger.error("OMNIX_PATH not set, cannot load prompts")
                return
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            logger.error("Path %s does not exist, cannot load prompts", path)
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for key, value in data.items():
            attr_name = key.lower()

            if hasattr(self, attr_name):
                setattr(self, attr_name, value)
                logger.debug("overriding prompt %s with %s", attr_name, value)
            else:
                self.extra_prompts[key] = value
        logger.info("Prompts loaded from %s", path)


PROMPTS = PromptRegistry()
PROMPTS.load_overrides()
