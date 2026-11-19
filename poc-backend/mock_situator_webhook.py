"""
Situator Web POC / Receptor de Webhooks PSIM (porta 8090)
Recebe ocorrências reais via POST /webhook, disponibiliza fila em /events,
permite tratamento pelo operador em /events/{id}/ack e serve um painel Web em /.
"""

import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="Situator Web POC - PSIM Event Receiver")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

received_events: list[dict] = []

SITUATOR_WEB_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Situator Web POC — Central de Ocorrências PSIM (Porta 8090)</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: "Segoe UI", Roboto, sans-serif; background: #090e1a; color: #f8fafc; min-height: 100vh; }
    header { background: #0f172a; border-bottom: 1px solid #1e293b; padding: 0.9rem 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem; }
    .brand { display: flex; align-items: center; gap: 0.7rem; font-weight: 700; font-size: 1.05rem; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399; }
    main { max-width: 1200px; margin: 0 auto; padding: 1.5rem; display: grid; gap: 1.25rem; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.9rem; }
    .stat { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 1rem; }
    .stat small { color: #94a3b8; font-size: 0.78rem; }
    .stat strong { display: block; font-size: 1.65rem; margin-top: 0.25rem; }
    .events-grid { display: grid; gap: 0.9rem; }
    .card { background: #0f172a; border: 1px solid #24324a; border-left: 4px solid #f87171; border-radius: 12px; padding: 1rem; display: grid; grid-template-columns: 180px 1fr auto; gap: 1rem; align-items: center; }
    .card.done { border-left-color: #34d399; opacity: 0.78; }
    .card img { width: 180px; height: 112px; object-fit: cover; border-radius: 8px; border: 1px solid #334155; background: #050811; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .b-open { background: rgba(248,113,113,0.16); color: #fca5a5; }
    .b-done { background: rgba(52,211,153,0.16); color: #6ee7b7; }
    button { background: #0284c7; color: #fff; border: 0; border-radius: 8px; padding: 0.55rem 0.95rem; font-weight: 600; cursor: pointer; }
    button.sec { background: #1e293b; border: 1px solid #334155; }
    @media (max-width: 720px) { .card { grid-template-columns: 1fr; } .card img { width: 100%; height: 180px; } }
  </style>
</head>
<body>
  <header>
    <div class="brand"><span class="dot"></span> Situator Web POC • Fila de Eventos PSIM (http://127.0.0.1:8090)</div>
    <div style="display:flex; gap:0.6rem; align-items:center;">
      <a href="http://127.0.0.1:5500/" style="color:#38bdf8; text-decoration:none; font-size:0.86rem;">← Voltar ao Painel Axioma Vision</a>
      <button class="sec" onclick="clearAll()">Limpar Fila</button>
    </div>
  </header>
  <main>
    <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:10px; padding:0.75rem 1rem; font-size:0.86rem; color:#cbd5e1;">
      <strong>Ambiente POC de Recepção PSIM:</strong> este console simula a fila web do Situator recebendo <code>POST /webhook</code> em tempo real. Em produção, aponte o Webhook para o servidor <strong>Situator Web corporativo da Seventh</strong> do cliente.
    </div>
    <div class="stats">
      <div class="stat"><small>Ocorrências Abertas</small><strong id="s-open" style="color:#f87171;">0</strong></div>
      <div class="stat"><small>Total Recebidas (Webhook)</small><strong id="s-total">0</strong></div>
      <div class="stat"><small>Endpoint Webhook Ativo</small><strong style="font-size:1rem; color:#34d399; margin-top:0.65rem;">POST /webhook</strong></div>
    </div>
    <div id="list" class="events-grid"></div>
  </main>
  <script>
    async function load() {
      const r = await fetch('/events');
      const d = await r.json();
      document.getElementById('s-open').textContent = d.open_count ?? 0;
      document.getElementById('s-total').textContent = d.count ?? 0;
      const list = document.getElementById('list');
      if (!d.events || !d.events.length) {
        list.innerHTML = '<div style="color:#94a3b8; padding:2rem; text-align:center; background:#0f172a; border:1px solid #1e293b; border-radius:12px;">Aguardando ocorrências em tempo real do backend IA (porta 8000)...</div>';
        return;
      }
      list.innerHTML = d.events.map(ev => {
        const done = ev.status === 'CONCLUIDA';
        const img = ev.evidenceFile ? `<img src="http://127.0.0.1:8000/recordings/${ev.evidenceFile}" alt="Evidência LGPD" />` : `<img alt="Sem miniatura" />`;
        return `
          <div class="card ${done ? 'done' : ''}">
            ${img}
            <div style="display:grid; gap:0.35rem;">
              <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
                <span class="badge ${done ? 'b-done' : 'b-open'}">${done ? 'CONCLUÍDA' : 'ALTA • ABERTA'}</span>
                <strong>${ev.eventCode || 'EVENTO'} — ${ev.account || ''} • ${ev.zone || ''}</strong>
              </div>
              <div style="color:#cbd5e1; font-size:0.9rem;">${ev.description || ''}</div>
              <div style="color:#94a3b8; font-size:0.78rem;">Arquivo: ${ev.videoFile || 'Live'} | Confiança: ${Math.round((ev.confidence || 0) * 100)}% | Recebido: ${ev.receivedAt || ''}</div>
            </div>
            <div>
              ${done ? '<span style="color:#34d399; font-weight:600;">Tratada ✓</span>' : `<button onclick="ack('${ev.id}')">Assumir &amp; Concluir</button>`}
            </div>
          </div>
        `;
      }).join('');
    }
    async function ack(id) { await fetch('/events/' + id + '/ack?status=CONCLUIDA', { method: 'POST' }); load(); }
    async function clearAll() { await fetch('/events/clear', { method: 'POST' }); load(); }
    setInterval(load, 2000);
    load();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def situator_web_console() -> str:
    return SITUATOR_WEB_HTML


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
