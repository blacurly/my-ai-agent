"""
AI Email Writing Agent
An intelligent agent that helps write, refine, and send emails using Claude API.
"""

import anthropic
import json
from typing import Any

# Initialize the Anthropic client
client = anthropic.Anthropic()
MODEL_ID = "claude-opus-4-8"

# Define the tools our email agent can use
tools = [
    {
        "name": "draft_email",
        "description": "Draft an email based on the topic, tone, and recipient. Returns a complete email draft.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "The recipient's name or email address"
                },
                "subject": {
                    "type": "string",
                    "description": "The email subject line"
                },
                "purpose": {
                    "type": "string",
                    "description": "The main purpose or topic of the email"
                },
                "tone": {
                    "type": "string",
                    "enum": ["formal", "casual", "friendly", "professional", "urgent"],
                    "description": "The desired tone of the email"
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Main points to include in the email"
                }
            },
            "required": ["recipient", "subject", "purpose", "tone"]
        }
    },
    {
        "name": "refine_email",
        "description": "Refine or improve an existing email draft with specific changes or improvements",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_content": {
                    "type": "string",
                    "description": "The current email content to refine"
                },
                "feedback": {
                    "type": "string",
                    "description": "Specific feedback or changes needed (e.g., 'make it shorter', 'add urgency', 'formal tone')"
                }
            },
            "required": ["email_content", "feedback"]
        }
    },
    {
        "name": "get_email_templates",
        "description": "Retrieve email templates for common scenarios",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["business", "follow_up", "complaint", "thank_you", "job_application"],
                    "description": "Category of email template"
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "check_email_quality",
        "description": "Analyze email for tone, clarity, grammar, and professionalism",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_content": {
                    "type": "string",
                    "description": "The email content to analyze"
                },
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What to check (e.g., 'grammar', 'clarity', 'professionalism', 'tone')"
                }
            },
            "required": ["email_content"]
        }
    }
]


def draft_email(recipient: str, subject: str, purpose: str, tone: str, key_points: list = None) -> str:
    """Simulate drafting an email"""
    key_points_text = "\n".join(f"- {point}" for point in (key_points or []))
    return f"""
EMAIL DRAFT:
To: {recipient}
Subject: {subject}
Tone: {tone}

Dear {recipient.split()[0] if ' ' in recipient else recipient},

I'm writing regarding {purpose.lower()}.

Key points:
{key_points_text if key_points_text else "- (No specific points provided)"}

I look forward to your response.

Best regards"""


def refine_email(email_content: str, feedback: str) -> str:
    """Simulate refining an email"""
    return f"""
REFINED EMAIL:
[Applied: {feedback}]

{email_content}

[Refinement suggestions completed. Review and make final adjustments as needed.]"""


def get_email_templates(category: str) -> str:
    """Return email templates based on category"""
    templates = {
        "business": "Subject: Meeting Request\n\nDear [Name],\n\nI would like to schedule a meeting to discuss [topic]...",
        "follow_up": "Subject: Follow-up on [Previous Topic]\n\nHi [Name],\n\nI wanted to follow up on our previous conversation...",
        "complaint": "Subject: Complaint Regarding [Issue]\n\nDear [Name],\n\nI am writing to formally lodge a complaint about...",
        "thank_you": "Subject: Thank You\n\nDear [Name],\n\nI wanted to express my sincere gratitude for...",
        "job_application": "Subject: Application for [Position]\n\nDear [Hiring Manager],\n\nI am writing to express my interest in the position..."
    }
    return templates.get(category, "Template not found")


def check_email_quality(email_content: str, criteria: list = None) -> str:
    """Simulate quality checking an email"""
    checks = criteria or ["grammar", "clarity", "professionalism"]
    feedback = f"\nEMAIL QUALITY ANALYSIS:\nChecking: {', '.join(checks)}\n\n"
    feedback += "✓ Grammar: Well-structured sentences\n"
    feedback += "✓ Clarity: Message is clear and concise\n"
    feedback += "✓ Professionalism: Appropriate tone for business\n"
    feedback += "\nOverall: Email is ready to send!"
    return feedback


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Process tool calls and return results"""
    if tool_name == "draft_email":
        return draft_email(**tool_input)
    elif tool_name == "refine_email":
        return refine_email(**tool_input)
    elif tool_name == "get_email_templates":
        return get_email_templates(**tool_input)
    elif tool_name == "check_email_quality":
        return check_email_quality(**tool_input)
    else:
        return f"Unknown tool: {tool_name}"


def run_email_agent(user_message: str) -> None:
    """
    Run the email agent with the given user message.
    The agent will use tools as needed to help with email writing.
    """
    print(f"\n{'='*60}")
    print(f"User Request: {user_message}")
    print(f"{'='*60}\n")
    
    messages = [{"role": "user", "content": user_message}]
    
    # System prompt for the email agent
    system_prompt = """You are an expert email writing assistant. Your goal is to help users write, 
refine, and improve emails. You have access to several tools to help with this task:
- draft_email: Create new email drafts
- refine_email: Improve existing emails
- get_email_templates: Provide templates for common scenarios
- check_email_quality: Analyze emails for quality and professionalism

Always provide helpful suggestions and be ready to iterate based on user feedback."""
    
    # Agentic loop
    while True:
        # Call Claude with tools
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages
        )
        
        print(f"Stop Reason: {response.stop_reason}")
        
        # Check if we're done
        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAssistant: {block.text}")
            break
        
        # Process tool calls if needed
        if response.stop_reason == "tool_use":
            # Add assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})
            
            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    print(f"Tool Called: {tool_name}")
                    print(f"Tool Input: {json.dumps(tool_input, indent=2)}")
                    
                    # Execute the tool
                    result = process_tool_call(tool_name, tool_input)
                    print(f"Tool Result:\n{result}\n")
                    
                    # Collect tool results
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason
            print(f"Unexpected stop reason: {response.stop_reason}")
            break


def main():
    """Example usage of the email agent"""
    
    # Example 1: Draft a business email
    print("\n" + "="*60)
    print("EXAMPLE 1: Drafting a Business Email")
    print("="*60)
    run_email_agent(
        "Help me draft a professional email to Sarah Johnson requesting a meeting "
        "to discuss the Q3 marketing strategy. I want it to be formal but friendly."
    )
    
    # Example 2: Refine an email
    print("\n" + "="*60)
    print("EXAMPLE 2: Refining an Email")
    print("="*60)
    run_email_agent(
        "I have this email draft:\n\n'Hi Tom, I wanted to talk about the project. "
        "Can we meet?' Can you help me make it more professional and detailed?"
    )
    
    # Example 3: Get a template and customize it
    print("\n" + "="*60)
    print("EXAMPLE 3: Using Email Templates")
    print("="*60)
    run_email_agent(
        "I need to send a follow-up email after our last conversation. "
        "Can you show me a template and help me customize it?"
    )


if __name__ == "__main__":
    main()
