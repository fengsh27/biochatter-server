from typing import Optional, Any
from flask import Flask, request
from dotenv import load_dotenv
from dotenv import load_dotenv
from src.conversation_manager import (
    get_conversation,
    has_conversation, 
    initialize_conversation
)
from pprint import pprint


load_dotenv()

# openai.api_type=os.environ["OPENAI_API_TYPE"]
# openai.api_base=os.environ["OPENAI_API_BASE"]
# openai.api_version=os.environ["OPENAI_API_VERSION"]
# openai.api_key=os.environ["OPENAI_API_KEY"]
# 
# response = openai.ChatCompletion.create(
  # engine=os.environ["OPENAI_DEPLOYMENT_NAME"],
  # messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"tell me about google bard"},{"role":"assistant","content":"Google has not created a product called \"Google Bard\". It's possible that you may have confused it with another product or there may be a misunderstanding about the name. Google does have a variety of tools and products related to writing, language processing, and artificial intelligence, but none of them are specifically named \"Google Bard\"."}],
  # temperature=0.7,
  # max_tokens=800,
  # top_p=0.95,
  # frequency_penalty=0,
  # presence_penalty=0,
  # stop=None)
# 
# from pprint import pprint
# pprint(response)
# 

load_dotenv()

app = Flask(__name__)

def get_params_from_json_body(json: Optional[Any], name: str, defaultVal: Optional[Any]) -> Optional[Any]:
    if not json:
        return defaultVal
    if name in json:
        return json[name]
    return defaultVal

@app.route('/v1/chat/completions', methods=['POST'])
def handle():
    auth = request.headers.get("Authorization")
    jsonBody = request.json
    sessionId = get_params_from_json_body(jsonBody, "session_id", defaultVal="")
    messages = get_params_from_json_body(jsonBody, "messages", defaultVal=[])
    model = get_params_from_json_body(jsonBody, "model", defaultVal="gpt-3.5-turbo")
    temperature = get_params_from_json_body(jsonBody, "temperature", defaultVal=0.7)
    presence_penalty = get_params_from_json_body(jsonBody, "presence_penalty", defaultVal=0)
    frequency_penalty = get_params_from_json_body(jsonBody, "frequency_penalty", defaultVal=0)
    top_p = get_params_from_json_body(jsonBody, "top_p", defaultVal=1)
    if not has_conversation(sessionId):
        initialize_conversation(
            session=sessionId,
            modelConfig={
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "top_p": top_p,
                "model": model,
                "auth": auth
            }
        )
    conversation = get_conversation(sessionId)
    try:
        (msg, usage) = conversation.chat(messages, auth)
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": msg}, "finish_reason": "stop"}], "usage": usage}
    except Exception as e:
        return {"error": str(e)}

