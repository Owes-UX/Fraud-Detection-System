import ulid

from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler


def main():
    # === KEYS FROM CHALLENGE PORTAL (hard-coded) ===
    LANGFUSE_PUBLIC_KEY = "pk-lf-42c15709-42a8-4522-8675-a5b1c91db0c9"
    LANGFUSE_SECRET_KEY = "sk-lf-2d0ba4ab-116e-445a-88c2-3d2ee1d5e304"
    LANGFUSE_HOST = "https://challenges.reply.com/langfuse"
    OPENROUTER_API_KEY = "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"
    TEAM_NAME = "Masala_Tech"

    print("DEBUG PUBLIC:", LANGFUSE_PUBLIC_KEY)
    print("DEBUG HOST:", LANGFUSE_HOST)

    # Initialize Langfuse client (mainly to validate keys)
    _ = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )

    # LangChain callback handler for Langfuse
    handler = CallbackHandler()

    # Unique session id required by the challenge
    session_id = f"{TEAM_NAME}-{ulid.new().str}"

    # LLM via OpenRouter (OpenAI-compatible)
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="openai/gpt-4.1-mini",
        callbacks=[handler],
    )

    config = {"metadata": {"langfuse_session_id": session_id}}

    @observe(name="trace_test_session", capture_input=True, capture_output=True)
    def run_test_calls():
        questions = [
            "Say 'hello' in one short sentence.",
            "In one sentence, explain what fraud detection is.",
            "Output a number between 0 and 1, like 0.42.",
        ]
        for q in questions:
            _ = llm.invoke(q, config=config)

    run_test_calls()

    print("Finished traced Langfuse session:", session_id)


if __name__ == "__main__":
    main()