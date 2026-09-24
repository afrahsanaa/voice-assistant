import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your answers are read aloud by a "
    "text-to-speech engine, so reply in plain sentences only: no markdown, "
    "no asterisks, no bullet or numbered lists, no emojis, no code blocks. "
    "Keep answers to one or two short sentences, and reply in the same "
    "language as the user."
)


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def create_client():
    load_dotenv()
    client = OpenAI(
        base_url=required_env("BASE_URL"),
        api_key=required_env("API_KEY"),
    )
    return client, required_env("LLM_MODEL")


def new_messages() -> list[ChatCompletionMessageParam]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def ask_llm(client, model_name, messages, text):
    messages.append({"role": "user", "content": text})
    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
    except AuthenticationError:
        messages.pop()
        print("Authentication failed: check API_KEY and BASE_URL.")
        return None
    except RateLimitError:
        messages.pop()
        print("Rate limit reached: please wait and try again.")
        return None
    except APIConnectionError:
        messages.pop()
        print("Could not connect to the API: check BASE_URL and your network.")
        return None
    except APIError as error:
        messages.pop()
        print(f"The API returned an error: {error}")
        return None

    assistant_text = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": assistant_text})
    return assistant_text


def main():
    client, llm_model = create_client()
    messages = new_messages()

    print("Chat started. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_text = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nChat ended.")
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            break

        try:
            assistant_text = ask_llm(client, llm_model, messages, user_text)
        except KeyboardInterrupt:
            print("\nChat interrupted.")
            break
        if assistant_text is None:
            continue

        print(f"Assistant: {assistant_text}")


if __name__ == '__main__':
    main()
