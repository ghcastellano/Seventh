# Backend POC — Antifurto em Mercados (D-Guard + YOLOv8-Pose + VLM + Situator)

API leve em **FastAPI** para processar clipes MP4 enviados pelo painel web da POC (ou capturados via RTSP `554` do D-Guard) e gerar alertas no formato **Webhook / API HTTP/JSON do Situator**.

## Rodando localmente ou em GPU (RunPod / Vast.ai)

```bash
cd poc-backend
python -m venv .venv
source .venv/bin/activate  # no Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Opcional: configurar VLM (Qwen2.5-VL via OpenRouter) e Webhook do Situator
export VLM_API_KEY="sua-chave-openrouter"
export SITUATOR_WEBHOOK_URL="http://seu-situator:8080/api/webhook"
export SITUATOR_TOKEN="seu-token"

uvicorn main:app --host 0.0.0.0 --port 8000
```

Depois, cole a URL do backend (ex.: `http://localhost:8000/analyze` ou sua URL pública do RunPod/Modal) na aba **Demo POC (Antifurto)** do painel web.
