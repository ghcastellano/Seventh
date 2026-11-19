"""
Situator Web POC / Receptor de Webhooks PSIM (porta 8090)
Recebe ocorrências reais via POST /webhook, registra desfecho operacional + ROI
(Furto Evitado R$, Atendimento Preventivo ou Falso Alarme) e serve painel Web em /.
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
  <title>Situator Web POC — Central de Ocorrências &amp; ROI da POC</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: "Segoe UI", Roboto, sans-serif; background: #090e1a; color: #f8fafc; min-height: 100vh; }
    header { background: #0f172a; border-bottom: 1px solid #1e293b; padding: 0.9rem 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem; }
    .brand { display: flex; align-items: center; gap: 0.7rem; font-weight: 700; font-size: 1.02rem; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399; }
    main { max-width: 1200px; margin: 0 auto; padding: 1.5rem; display: grid; gap: 1.25rem; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 0.9rem; }
    .stat { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 1rem; }
    .stat small { color: #94a3b8; font-size: 0.76rem; }
    .stat strong { display: block; font-size: 1.5rem; margin-top: 0.25rem; }
    .events-grid { display: grid; gap: 0.9rem; }
    .card { background: #0f172a; border: 1px solid #24324a; border-left: 4px solid #f87171; border-radius: 12px; padding: 1rem; display: grid; grid-template-columns: 170px 1fr auto; gap: 1rem; align-items: center; }
    .card.done { border-left-color: #34d399; opacity: 0.86; }
    .card.fp { border-left-color: #fbbf24; opacity: 0.78; }
    .card img { width: 170px; height: 108px; object-fit: cover; border-radius: 8px; border: 1px solid #334155; background: #050811; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
    .b-open { background: rgba(248,113,113,0.16); color: #fca5a5; }
    .b-done { background: rgba(52,211,153,0.16); color: #6ee7b7; }
    .b-fp { background: rgba(251,191,36,0.16); color: #fde68a; }
    .actions { display: flex; flex-direction: column; gap: 0.4rem; min-width: 170px; }
    button { background: #0284c7; color: #fff; border: 0; border-radius: 8px; padding: 0.45rem 0.75rem; font-size: 0.76rem; font-weight: 600; cursor: pointer; }
    button.ok { background: #059669; }
    button.warn { background: #334155; color: #fde68a; }
    button.sec { background: #1e293b; border: 1px solid #334155; }
    @media (max-width: 780px) { .card { grid-template-columns: 1fr; } .card img { width: 100%; height: 180px; } .actions { flex-direction: row; flex-wrap: wrap; } }
  </style>
</head>
<body>
  <header>
    <div class="brand"><span class="dot"></span> Situator Web POC • Auditoria PSIM &amp; ROI da POC</div>
    <div style="display:flex; gap:0.6rem; align-items:center;">
      <a href="http://127.0.0.1:5500/" style="color:#38bdf8; text-decoration:none; font-size:0.84rem;">← Voltar ao Painel</a>
      <button class="sec" onclick="clearAll()">Limpar Fila</button>
    </div>
  </header>
  <main>
    <div class="stats">
      <div class="stat"><small>Ocorrências Abertas</small><strong id="s-open" style="color:#f87171;">0</strong></div>
      <div class="stat"><small>Furtos Recuperados</small><strong id="s-rec" style="color:#34d399;">0</strong></div>
      <div class="stat"><small>Valor Recuperado (ROI POC)</small><strong id="s-brl" style="color:#38bdf8;">R$ 0</strong></div>
      <div class="stat"><small>Taxa de Assertividade</small><strong id="s-prec" style="color:#e2e8f0;">100%</strong></div>
    </div>
    <div id="list" class="events-grid"></div>
  </main>
  <script>
    async function load() {
      const r = await fetch('events');
      const d = await r.json();
      document.getElementById('s-open').textContent = d.open_count ?? 0;
      document.getElementById('s-rec').textContent = d.recovered_count ?? 0;
      document.getElementById('s-brl').textContent = 'R$ ' + Math.round(d.recovered_total_brl || 0);
      document.getElementById('s-prec').textContent = (d.precision_pct ?? 100) + '%';
      const list = document.getElementById('list');
      if (!d.events || !d.events.length) {
        list.innerHTML = '<div style="color:#94a3b8; padding:2rem; text-align:center; background:#0f172a; border:1px solid #1e293b; border-radius:12px;">Aguardando ocorrências em tempo real via Webhook...</div>';
        return;
      }
      const apiBase = location.pathname.startsWith('/situator') ? '..' : 'http://127.0.0.1:8000';
      list.innerHTML = d.events.map(ev => {
        const isDone = ev.status === 'CONCLUIDA';
        const isFp = ev.outcome === 'FALSO_ALARME';
        const img = ev.evidenceFile ? `<img src="${apiBase}/recordings/${ev.evidenceFile}" alt="Evidência LGPD" />` : `<img alt="Sem miniatura" />`;
        return `
          <div class="card ${isFp ? 'fp' : (isDone ? 'done' : '')}">
            ${img}
            <div style="display:grid; gap:0.3rem;">
              <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
                <span class="badge ${isFp ? 'b-fp' : (isDone ? 'b-done' : 'b-open')}">${isDone ? (ev.outcomeLabel || 'CONCLUÍDA') : 'ALTA • ABERTA'}</span>
                <strong>${ev.eventCode || 'EVENTO'} — ${ev.account || ''} • ${ev.zone || ''}</strong>
              </div>
              <div style="color:#cbd5e1; font-size:0.88rem;">${ev.description || ''}</div>
              <div style="color:#94a3b8; font-size:0.78rem;">POP: ${ev.recommendedAction || ''}</div>
              <div style="color:#64748b; font-size:0.75rem;">Clipe: ${ev.videoFile || 'Live'} | Score: ${Math.round((ev.confidence || 0) * 100)}% | Recebido: ${ev.receivedAt || ''}</div>
            </div>
            <div class="actions">
              ${isDone
                ? `<span style="color:${isFp ? '#fde68a' : '#34d399'}; font-weight:600; font-size:0.82rem;">${ev.outcomeLabel || 'Encerrada ✓'}</span>`
                : `<button class="ok" onclick="ack('${ev.id}','RECUPERADO',120)">✓ Recuperado (+R$ 120)</button>
                   <button onclick="ack('${ev.id}','ATENDIMENTO',0)">POP Atendimento</button>
                   <button class="warn" onclick="ack('${ev.id}','FALSO_ALARME',0)">Marcar Falso Alarme</button>`}
            </div>
          </div>
        `;
      }).join('');
    }
    async function ack(id, outcome, brl) {
      await fetch(`events/${id}/ack?status=CONCLUIDA&outcome=${outcome}&recovered_brl=${brl}`, { method: 'POST' });
      load();
    }
    async function clearAll() { await fetch('events/clear', { method: 'POST' }); load(); }
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
        "outcome": "",
        "outcomeLabel": "",
        "recoveredBrl": 0.0,
        "receivedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    received_events.insert(0, event)
    return {"status": "received", "event_id": event["id"], "total_events": len(received_events)}


@app.get("/events")
def list_events() -> dict:
    open_count = sum(1 for ev in received_events if ev.get("status") != "CONCLUIDA")
    rec_count = sum(1 for ev in received_events if ev.get("outcome") == "RECUPERADO")
    rec_brl = sum(float(ev.get("recoveredBrl") or 0) for ev in received_events)
    handled = [ev for ev in received_events if ev.get("status") == "CONCLUIDA"]
    fp_count = sum(1 for ev in handled if ev.get("outcome") == "FALSO_ALARME")
    precision = round(((len(handled) - fp_count) / len(handled)) * 100) if handled else 100
    return {
        "count": len(received_events),
        "open_count": open_count,
        "recovered_count": rec_count,
        "recovered_total_brl": rec_brl,
        "precision_pct": precision,
        "events": received_events[:25],
    }


@app.post("/events/{event_id}/ack")
async def acknowledge_event(
    event_id: str,
    status: str = "CONCLUIDA",
    outcome: str = "RECUPERADO",
    recovered_brl: float = 0.0,
) -> dict:
    labels = {
        "RECUPERADO": f"Recuperado (R$ {int(recovered_brl)}) ✓",
        "ATENDIMENTO": "Atendimento Preventivo ✓",
        "FALSO_ALARME": "Falso Alarme Registrado",
    }
    for ev in received_events:
        if ev.get("id") == event_id:
            ev["status"] = status
            ev["outcome"] = outcome
            ev["outcomeLabel"] = labels.get(outcome, outcome)
            ev["recoveredBrl"] = recovered_brl
            ev["handledAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return {"ok": True, "event": ev}
    raise HTTPException(status_code=404, detail="Ocorrência não encontrada.")


@app.post("/events/clear")
def clear_events() -> dict:
    received_events.clear()
    return {"ok": True, "count": 0}
