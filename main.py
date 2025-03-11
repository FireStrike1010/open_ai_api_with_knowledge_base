import uvicorn
from rag_api.app import app


if __name__ == '__main__':
    uvicorn.run(app=app, reload=False)