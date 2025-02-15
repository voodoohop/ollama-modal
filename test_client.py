from openai import OpenAI

# Initialize the client with the Modal deployment URL
client = OpenAI(
    base_url="https://pollinations--ollama-api-fastapi-app.modal.run/v1",
    api_key="not-needed"  # The API key isn't used but required by the client
)

def test_chat():
    print("\nTesting chat completion...")
    
    # Test with streaming
    print("\nStreaming response:")
    response = client.chat.completions.create(
        model="mistral-small",
        messages=[
            {"role": "user", "content": "What is the capital of France and what's special about it?"}
        ],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")
    
    # Test without streaming
    print("\nNon-streaming response:")
    response = client.chat.completions.create(
        model="mistral-small",
        messages=[
            {"role": "user", "content": "What is the capital of France and what's special about it?"}
        ],
        stream=False
    )
    
    print(response.choices[0].message.content)
    print()

if __name__ == "__main__":
    test_chat()
