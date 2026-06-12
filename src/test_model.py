from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

for model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]:
    try:
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))
        resp = llm.invoke("Say hello")
        print(f"{model}: Working")
        break
    except Exception as e:
        print(f"{model}: {str(e)[:80]}")