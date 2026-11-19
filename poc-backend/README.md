# Backend POC — Antifurto em Mercados (D-Guard + YOLOv8-Pose + VLM + Situator)

Stack Python completo para validar a POC ponta a ponta (sem precisar de câmera física no início):

- `main.py` — API FastAPI (`POST /analyze`) usada pela aba **Demo POC (Antifurto)** do site na Vercel.
- `rtsp_worker.py` — Worker em tempo real que lê o **RTSP (porta 554)** do D-Guard (ou simulador MediaMTX/FFmpeg), roda **YOLOv8-Pose**, valida com **VLM** e envia **Webhook para o Situator**.
- `mock_situator_webhook.py` — Receptor de teste simulando o Situator na porta `8090`.
- `test_e2e.py` — Script de teste automatizado que gera um MP4 sintético e valida o `/analyze`.

## Como rodar e testar (passo a passo)

```bash
cd poc-backend
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: Simulador do Webhook do Situator (opcional se já tiver Situator real)
uvicorn mock_situator_webhook:app --port 8090

# Terminal 2: API da POC (para usar com a aba Demo POC do site)
export SITUATOR_WEBHOOK_URL="http://127.0.0.1:8090/webhook"
# Opcional: export VLM_API_KEY="sua-chave-openrouter"
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 3: Teste E2E rápido
python test_e2e.py
```

## Para testar com RTSP simulado (sem câmera física)

1. Baixe o **MediaMTX** e rode na porta `8554`.
2. Envie um vídeo MP4 em loop como se fosse uma câmera do D-Guard:
   `ffmpeg -re -stream_loop -1 -i seu_video.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/corredor03`
3. Rode o worker em tempo real:
   `python rtsp_worker.py`
