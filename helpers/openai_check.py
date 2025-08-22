from openai import OpenAI
import os, sys

def check_key(model: str = "gpt-4o-mini"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY set")
        return 1
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":"ping"}], max_tokens=4)
        print("OK", resp.choices[0].message.content)
        return 0
    except Exception as e:
        print("Error", e)
        return 2

if __name__ == "__main__":
    sys.exit(check_key())
