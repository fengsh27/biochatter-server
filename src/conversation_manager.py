import os
from datetime import datetime
import logging
from typing import Optional, Dict, List, Any
from biochatter.llm_connect import (
    AzureGptConversation, 
    GptConversation, 
    Conversation
)
from pprint import pprint
from src.constants import (
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_API_TYPE,
    OPENAI_API_VERSION,
    OPENAI_DEPLOYMENT_NAME,
    OPENAI_MODEL
)

logger = logging.getLogger(__name__)

defaultModelConfig = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "sendMemory": True,
    "historyMessageCount": 4,
    "compressMessageLengthThreshold": 2000,
}

MAX_AGE = 3*24*3600*1000 # 3 days

def parse_api_key(bearToken: str) -> Dict:
    bearToken = bearToken.strip()
    bearToken.replace("bearer", "")
    return bearToken

class SessionData:
    def __init__(self, session: str, modelConfig: Dict, chatter: Optional[GptConversation]):
        self.modelConfig = modelConfig
        self.chatter = chatter
        self.sessionId = session

        self.createdAt = int(datetime.now().timestamp()*1000) # in milliseconds
        self.refreshedAt = self.createdAt
        self.maxAge =  MAX_AGE

    def chat(self, messages: Optional[List[str]], authKey: Optional[str]):
        if self.chatter is None:
            return
        if not messages or len(messages) == 0:
            return
        if not isinstance(self.chatter, AzureGptConversation): # chatter is instance of GptConversation
            import openai
            if not openai.api_key:
                if not authKey:
                    return False
                self.chatter.set_api_key(parse_api_key(authKey))
        text = messages[-1]["content"]
        messages = messages[:-1]
        pprint(messages)
        self._setup_messages(messages)
        try:
            (msg, usage, _) = self.chatter.query(text)
            return (msg, usage)
        except Exception as e:
            logger.error(e)
            raise e
    
    def _setup_messages(self, openai_msgs: List[Any]):
        if self.chatter is None:
            return False
        self.chatter.messages = []
        for msg in openai_msgs:
            if msg["role"] == "system":
                self.chatter.append_system_message(msg["content"])
            elif msg["role"] == "assistant":
                self.chatter.append_ai_message(msg["content"])
            elif msg["role"] == "user":
                self.chatter.append_user_message(msg["content"])

conversationsDict = {}

def initialize_conversation(session: str, modelConfig: dict):
    if os.environ[OPENAI_API_TYPE] == "azure":
        chatter = AzureGptConversation(
            deployment_name=os.environ[OPENAI_DEPLOYMENT_NAME],
            model_name=os.environ[OPENAI_MODEL],
            prompts={},
            version=os.environ[OPENAI_API_VERSION],
            base=os.environ[OPENAI_API_BASE],
        )
        chatter.set_api_key(os.environ[OPENAI_API_KEY])
        conversationsDict[session] = SessionData( session, modelConfig, chatter)
    else:
        chatter = GptConversation("gpt-3.5-turbo", prompts={})
        conversationsDict[session] = SessionData(
            session=session,
            modelConfig=modelConfig,
            chatter=chatter
        )
def has_conversation(session: str) -> bool:
    return session in conversationsDict
def get_conversation(session: str) -> Optional[SessionData]:
    if not session in conversationsDict:
        initialize_conversation(session, defaultModelConfig.copy())
    return conversationsDict[session]
    
def remove_conversation(session: str):
    if not session in conversationsDict:
        return
    del conversationsDict[session]



