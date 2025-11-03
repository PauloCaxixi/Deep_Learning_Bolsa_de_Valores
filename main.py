from fastapi import FastAPI
from app.interfaces.routes import router as api_router  # <— note the "app."

def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router, prefix="/v1")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
