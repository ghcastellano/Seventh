"""
Simulador local do receptor de Webhooks do Situator (porta 8090) com CORS habilitado
para visualização em tempo real no painel Demo POC.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mock Situator Webhook Receiver")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

received_events: list[dict] = []


@app.post("/webhook")
async def receive_webhook(request: Request) -> dict:
    payload = await request.json()
    received_events.insert(0, payload)
    print("\n[SITUATOR MOCK] Nova ocorrência recebida:")
    print(payload)
    return {"status": "received", "total_events": len(received_events)}


@app.get("/events")
def list_events() -> dict:
    return {"count": len(received_events), "events": received_events[:20]}
