"""
Seventh AI — Worker RTSP em Tempo Real (D-Guard RTSP :554 -> YOLOv8-Pose -> Juiz VLM -> Situator Webhook)
Conecta em uma câmera ou servidor D-Guard via RTSP (ou simulador MediaMTX/FFmpeg),
analisa pose em tempo real e dispara ocorrência JSON para o Situator quando detecta ocultação suspeita.
"""

import asyncio
import base64
import collections
import os
import time

import cv2
import httpx
from ultralytics import YOLO

RTSP_URL = os.getenv("DGUARD_RTSP_URL", "rtsp://127.0.0.1:8554/corredor03")
SITUATOR_WEBHOOK_URL = os.getenv("SITUATOR_WEBHOOK_URL", "http://127.0.0.1:8090/webhook")
SITUATOR_TOKEN = os.getenv("SITUATOR_TOKEN", "demo-token-situator")
ACCOUNT_NAME = os.getenv("SITUATOR_ACCOUNT", "Mercado Piloto 01")
ZONE_NAME = os.getenv("SITUATOR_ZONE", "Corredor 03 - Bebidas")
VLM_API_URL = os.getenv("VLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
VLM_API_KEY = os.getenv("VLM_API_KEY", "")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")

POSE_THRESHOLD = float(os.getenv("POSE_THRESHOLD", "0.68"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "15"))


def score_pose_frame(pose_model: YOLO, frame) -> float:
    results = pose_model(frame, verbose=False)
    best = 0.0
    for res in results:
        if res.keypoints is None or res.keypoints.xy is None:
            continue
        for kpts in res.keypoints.xy:
            if len(kpts) < 13:
                continue
            l_wrist, r_wrist = kpts[9], kpts[10]
            l_hip, r_hip = kpts[11], kpts[12]
            l_sh = kpts[5]
            torso_h = max(float(abs(l_hip[1] - l_sh[1])), 40.0)
            dist_l = float(((l_wrist[0] - l_hip[0]) ** 2 + (l_wrist[1] - l_hip[1]) ** 2) ** 0.5)
            dist_r = float(((r_wrist[0] - r_hip[0]) ** 2 + (r_wrist[1] - r_hip[1]) ** 2) ** 0.5)
            ratio = min(dist_l, dist_r) / torso_h
            if ratio < 0.42:
                best = max(best, min(0.95, 1.0 - ratio))
    return round(best, 2)


async def confirm_with_vlm(frame) -> tuple[bool, str]:
    if not VLM_API_KEY:
        return True, "Modo sem VLM_API_KEY: confirmado apenas por score de pose local."
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode("utf-8")
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Corredor {ZONE_NAME}. Há indício visual de ocultação de produto na roupa/bolsa? Responda curto.",
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": 120,
    }
    headers = {"Authorization": f"Bearer {VLM_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(VLM_API_URL, json=payload, headers=headers)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return True, txt


async def send_situator_webhook(score: float, reason: str) -> None:
    payload = {
        "token": SITUATOR_TOKEN,
        "eventCode": "AI_SHOPLIFTING_SUSPECT",
        "priority": "ALTA" if score >= 0.78 else "MEDIA",
        "account": ACCOUNT_NAME,
        "zone": ZONE_NAME,
        "confidence": score,
        "description": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(SITUATOR_WEBHOOK_URL, json=payload)
        print(f"[SITUATOR] Webhook enviado ({r.status_code}): {payload['eventCode']} score={score}")


async def main() -> None:
    print(f"[WORKER] Conectando ao RTSP: {RTSP_URL}")
    pose_model = YOLO(os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt"))
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o stream RTSP: {RTSP_URL}")

    ring = collections.deque(maxlen=75)  # ~5s a 15 FPS
    last_alert_ts = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.5)
            continue
        ring.append(frame)

        score = score_pose_frame(pose_model, frame)
        now = time.time()
        if score >= POSE_THRESHOLD and (now - last_alert_ts) >= COOLDOWN_SEC:
            last_alert_ts = now
            ok, reason = await confirm_with_vlm(frame)
            if ok:
                await send_situator_webhook(score, reason)


if __name__ == "__main__":
    asyncio.run(main())
