import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

class AIAgent:
    def __init__(self):
        self.conversation_history = []
    
    def chat(self, user_message):
        """Send a message and get a response from the AI agent."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are a helpful AI assistant. Answer questions concisely and accurately.",
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def run(self):
        """Run the agent in interactive mode."""
        print("AI Agent started! Type 'quit' to exit.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = self.chat(user_input)
            print(f"\nAgent: {response}\n")

if __name__ == "__main__":
    agent = AIAgent()
    agent.run()