"""
Seventh AI — Backend POC de Detecção de Furtos em Mercados (YOLOv8-Pose + Juiz VLM + Webhook Situator)
Executa análise em vídeo MP4 enviado pelo painel web ou stream RTSP do D-Guard (porta 554).
"""

import base64
import os
import tempfile
import time
from typing import Any

import cv2
import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="Seventh AI - Shoplifting POC API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega modelo leve de pose (baixa automaticamente na 1a execução)
pose_model = YOLO(os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt"))

VLM_API_URL = os.getenv("VLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
VLM_API_KEY = os.getenv("VLM_API_KEY", "")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")
SITUATOR_WEBHOOK_URL = os.getenv("SITUATOR_WEBHOOK_URL", "")
SITUATOR_TOKEN = os.getenv("SITUATOR_TOKEN", "")


def _extract_pose_suspicion(video_path: str, max_frames: int = 90) -> dict[str, Any]:
    """
    Analisa frames do vídeo com YOLOv8-Pose e calcula um score heurístico inicial
    baseado na proximidade dos pulsos (keypoints 9 e 10) em relação ao quadril/tronco (11 e 12).
    Na produção, este bloco é substituído/reforçado pelo classificador temporal ST-GCN (PoseLift).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, int(fps // 5))  # amostra ~5 FPS

    frame_idx = 0
    analyzed = 0
    suspicious_frames = 0
    best_frame_b64 = None
    best_score = 0.0

    while cap.isOpened() and analyzed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        results = pose_model(frame, verbose=False)
        frame_score = 0.0

        for res in results:
            if res.keypoints is None or res.keypoints.xy is None:
                continue
            for kpts in res.keypoints.xy:
                if len(kpts) < 13:
                    continue
                # Keypoints COCO: 9=pulso esq, 10=pulso dir, 11=quadril esq, 12=quadril dir, 5/6=ombros
                l_wrist, r_wrist = kpts[9], kpts[10]
                l_hip, r_hip = kpts[11], kpts[12]
                l_sh, r_sh = kpts[5], kpts[6]

                torso_h = max(float(abs(l_hip[1] - l_sh[1])), 40.0)
                dist_l = float(((l_wrist[0] - l_hip[0]) ** 2 + (l_wrist[1] - l_hip[1]) ** 2) ** 0.5)
                dist_r = float(((r_wrist[0] - r_hip[0]) ** 2 + (r_wrist[1] - r_hip[1]) ** 2) ** 0.5)

                min_ratio = min(dist_l, dist_r) / torso_h
                if min_ratio < 0.42:
                    frame_score = max(frame_score, min(0.95, 1.0 - min_ratio))

        if frame_score >= 0.60:
            suspicious_frames += 1
        if frame_score > best_score:
            best_score = frame_score
            _, buf = cv2.imencode(".jpg", frame)
            best_frame_b64 = base64.b64encode(buf).decode("utf-8")

        analyzed += 1
        frame_idx += 1

    cap.release()
    overall_score = round(min(0.98, best_score * 0.7 + min(0.3, suspicious_frames * 0.05)), 2)
    return {
        "frames_analyzed": analyzed,
        "suspicious_frames": suspicious_frames,
        "pose_score": overall_score,
        "best_frame_b64": best_frame_b64,
    }


async def _ask_vlm_judge(frame_b64: str | None, corridor: str) -> dict[str, Any]:
    """Envia o melhor frame suspeito para o VLM (ex.: Qwen2.5-VL) confirmar intenção de ocultação."""
    if not VLM_API_KEY or not frame_b64:
        return {
            "used_vlm": False,
            "verdict": "SIMULADO (configure VLM_API_KEY para inferência real)",
            "reason": "Gesto com mão próxima à região da cintura/bolso detectado pelo YOLOv8-Pose.",
        }

    prompt = (
        f"Você é um auditor de prevenção a perdas no corredor '{corridor}' de um mercado. "
        "Analise a imagem e responda em JSON curto com as chaves 'suspeita' (true/false) e 'motivo' (1 frase): "
        "há indício visual de a pessoa estar escondendo um produto no bolso, roupa ou bolsa?"
    )
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                ],
            }
        ],
        "max_tokens": 180,
    }
    headers = {"Authorization": f"Bearer {VLM_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(VLM_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return {"used_vlm": True, "verdict": "CONFIRMADO_VLM", "reason": text}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "yolov8-pose"}


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    account: str = Form("Mercado Piloto 01"),
    zone: str = Form("Corredor 03 - Bebidas"),
) -> dict[str, Any]:
    t0 = time.time()
    suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pose_res = _extract_pose_suspicion(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    vlm_res = await _ask_vlm_judge(pose_res["best_frame_b64"], zone) if pose_res["pose_score"] >= 0.65 else {
        "used_vlm": False,
        "verdict": "DESCARTADO_POR_POSE",
        "reason": "Score de pose abaixo do limiar (0.65); economizou chamada ao VLM.",
    }

    situator_payload = {
        "token": SITUATOR_TOKEN or "demo-token-situator",
        "eventCode": "AI_SHOPLIFTING_SUSPECT",
        "priority": "ALTA" if pose_res["pose_score"] >= 0.75 else "MEDIA",
        "account": account,
        "zone": zone,
        "confidence": pose_res["pose_score"],
        "description": vlm_res["reason"],
        "recommendedAction": "Abordagem preventiva de atendimento ao cliente ('Posso ajudar?')",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    webhook_sent = False
    if SITUATOR_WEBHOOK_URL and pose_res["pose_score"] >= 0.70:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(SITUATOR_WEBHOOK_URL, json=situator_payload)
            webhook_sent = True

    return {
        "elapsed_ms": int((time.time() - t0) * 1000),
        "pose_analysis": {
            "frames_analyzed": pose_res["frames_analyzed"],
            "suspicious_frames": pose_res["suspicious_frames"],
            "pose_score": pose_res["pose_score"],
        },
        "vlm_judge": vlm_res,
        "situator_webhook_sent": webhook_sent,
        "situator_payload": situator_payload,
    }
