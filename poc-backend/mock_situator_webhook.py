"""
Receptor local do Situator PSIM (porta 8090) com fila operacional real,
reconhecimento/tratamento de ocorrências pelo operador e CORS habilitado.
"""

import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Situator PSIM Event Receiver (Local POC)")
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
    event = {
        "id": str(uuid.uuid4())[:8],
        "status": "ABERTA",
        "receivedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    received_events.insert(0, event)
    print("\n[SITUATOR 8090] Nova ocorrência recebida:")
    print(f"  ID={event['id']} | {event.get('eventCode')} | Zona={event.get('zone')} | Conf={event.get('confidence')}")
    return {"status": "received", "event_id": event["id"], "total_events": len(received_events)}


@app.get("/events")
def list_events() -> dict:
    open_count = sum(1 for ev in received_events if ev.get("status") != "CONCLUIDA")
    return {"count": len(received_events), "open_count": open_count, "events": received_events[:25]}


@app.post("/events/{event_id}/ack")
async def acknowledge_event(event_id: str, status: str = "CONCLUIDA") -> dict:
    for ev in received_events:
        if ev.get("id") == event_id:
            ev["status"] = status
            ev["handledAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return {"ok": True, "event": ev}
    raise HTTPException(status_code=404, detail="Ocorrência não encontrada.")


@app.post("/events/clear")
def clear_events() -> dict:
    received_events.clear()
    return {"ok": True, "count": 0}
