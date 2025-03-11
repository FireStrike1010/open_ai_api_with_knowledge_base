from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from datetime import datetime
from dotenv import dotenv_values
from httpx import HTTPStatusError

from .llm import OpenAIModel
from .models import StateResponse, LLMResponse, LLMRequest
from .utils import get_state, LoggerManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup_time = datetime.now()
    app.state.config = dotenv_values('.env')
    app.state.logger = LoggerManager(app.state.config.get('LOG_FOLDER'))
    app.state.llm = OpenAIModel(
        xml_folder=app.state.config.get('DATA_FOLDER'),
        api_key=app.state.config.get('OPENAI_API_KEY'),
        model_name=app.state.config.get('OPENAI_MODEL'),
        embedding_model_name=app.state.config.get('EMBEDDING_MODEL'),
        cache_file=app.state.config.get('CACHE_FILE'),
        system_prompt=app.state.config.get('SYSTEM_PROMPT')
    )
    app.state.knowledge_base_files = []
    if await app.state.llm.check_openai_api():
        try:
            app.state.knowledge_base_files = await app.state.llm.load_and_vectorize_knowledge()
        except FileNotFoundError as e:
            app.state.logger.logger.error(str(e))
        except HTTPStatusError as e:
            app.state.logger.logger.error(str(e))
    else:
        app.state.logger.logger.error('Unable to connect to Open AI api, check internet connection or OPENAI_API_KEY')
    yield


app = FastAPI(lifespan=lifespan)

@app.get('/', response_model=StateResponse)
async def home(G = Depends(get_state)):
    open_ai_status = 'online' if G.llm.check_openai_api() else 'offline'
    response = StateResponse(
        api_status='online',
        startup_time=G.startup_time,
        open_ai_status=open_ai_status,
        content_files=G.knowledge_base_files,
        model_name=G.config.get('OPENAI_MODEL'))
    return response

@app.post('/llm/', response_model=LLMResponse)
async def ask(data: LLMRequest, G = Depends(get_state)):
    try:
        open_ai_response = await G.llm.generate_response(
            query=data.query,
            top_k=data.top_k,
            top_p=data.top_p,
            temperature=data.temperature)
    except HTTPStatusError as e:
        G.logger.logger.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    return LLMResponse(answer=open_ai_response)