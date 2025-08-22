#!/usr/bin/env python3
"""
check_openai_key.py - Simple script to verify your OpenAI API key
Usage:
  setx OPENAI_API_KEY "sk-..."   # (Windows PowerShell, run once)
  python check_openai_key.py
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Please install the package first: pip install openai")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment.")
        print("   On Windows PowerShell, set it with:")
        print('   setx OPENAI_API_KEY "sk-..."')
        return

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",   # lightweight model
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ]
        )
        answer = response.choices[0].message.content
        print("✅ API key works! Response:", answer)
    except Exception as e:
        print("❌ API request failed.")
        print("Error:", e)

if __name__ == "__main__":
    main()
