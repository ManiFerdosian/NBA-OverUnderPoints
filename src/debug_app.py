from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Debug App")

@app.get("/home", response_class=HTMLResponse)
async def root():
    return "<h1>Hello from debug app</h1>"

@app.get("/health")
async def health():
    return {"status": "ok"}
