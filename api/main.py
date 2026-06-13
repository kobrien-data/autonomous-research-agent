from fastapi import FastAPI

app = FastAPI(title="Autonomous Research Agent API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research")
def research(query: str):
    return {"status": "stub", "query": query, "result": None}
