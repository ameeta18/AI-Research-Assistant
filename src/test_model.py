from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import get_settings

api_key = get_settings().require_google_api_key()

for model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]:
    try:
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        resp = llm.invoke("Say hello")
        print(f"{model}: Working")
        break
    except Exception as e:
        print(f"{model}: {str(e)[:80]}")
