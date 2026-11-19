"""
Simulador local do receptor de Webhooks do Situator (porta 8090).
Use para validar o fluxo ponta a ponta quando ainda não houver um servidor Situator configurado.
"""

from fastapi import FastAPI, Request

app = FastAPI(title="Mock Situator Webhook Receiver")
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
