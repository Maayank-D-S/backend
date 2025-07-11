# chat_ramvan_live.py

from bot import generate_response  # Replace with your actual filename (no .py)


def main():
    print("🟢 FireflyAssistant\nType 'exit' to quit.\n")

    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting chat. Have a great day!")
            break

        # Add latest user message to history
        history.append({"role": "user", "content": user_input})

        # Get response
        response = generate_response("Ramvan Villas", history, False)

        # Add bot response to history
        history.append({"role": "assistant", "content": response["text"]})

        # Display bot response
        print("\nBot:", response["text"])
        # if response["image_url"]:
        #     print("📸 Image URL:", response["image_url"])
        # print()


if __name__ == "__main__":
    main()
