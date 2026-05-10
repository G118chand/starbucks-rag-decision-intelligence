"""LLM factory — returns a configured chat model based on the LLM_PROVIDER env var."""

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def get_llm():
    """Instantiate and return the configured LLM.

    Reads ``LLM_PROVIDER`` from the environment (default ``"openai"``) and
    constructs the appropriate LangChain chat model.

    Supported providers:
    - ``openai``  — ``ChatOpenAI`` backed by GPT-3.5-turbo via ``OPENAI_API_KEY``.
    - ``bedrock`` — ``ChatBedrock`` backed by Claude 3 Sonnet on AWS Bedrock.

    Returns:
        A LangChain chat model that implements the ``Runnable`` interface.

    Raises:
        ValueError: If the required credentials are absent or the provider is unknown.
    """
    load_dotenv()
    provider: str = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    logger.info("LLM provider: %s", provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your-openai-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY is not set or still contains the placeholder value. "
                "Edit .env and add your real key, or set LLM_PROVIDER=bedrock."
            )
        logger.info("Initialising ChatOpenAI (gpt-3.5-turbo)")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1500,
            openai_api_key=api_key,
        )

    if provider == "bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError as exc:
            raise ImportError(
                "langchain-aws is required for the Bedrock provider. "
                "Install it with: pip install langchain-aws"
            ) from exc

        region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        logger.info("Initialising ChatBedrock (claude-3-sonnet) in region %s", region)
        return ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=region,
            model_kwargs={"max_tokens": 1500, "temperature": 0},
        )

    if provider == "mock":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        logger.info("Initialising FakeListChatModel (mock — no API key required)")
        mock_answer = (
            "According to [regional_summary.csv], the Pacific Northwest region posted "
            "9.5% comparable store sales growth in Q4 2024, driven by Reserve Roastery "
            "traffic and cold beverage attachment rates of 61%. Drive-thru average service "
            "time improved to 198 seconds from 217 seconds year-over-year.\n\n"
            "According to [sales_data.csv], weekly revenue in the Pacific Northwest "
            "averages $28,000–$42,000 per store depending on format, with Drive-Thru+Cafe "
            "locations generating the highest volume.\n\n"
            "Recommendations:\n"
            "1. Accelerate cold beverage menu expansion to capitalise on the 61% attachment rate.\n"
            "2. Replicate the drive-thru speed improvements (198s) across underperforming stores.\n"
            "3. Evaluate opening 3–5 additional Reserve format locations in Seattle and Portland "
            "to capture premium ticket growth."
        )
        return FakeListChatModel(responses=[mock_answer])

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Supported values: 'openai', 'bedrock', 'mock'."
    )
