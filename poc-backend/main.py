"""
Seventh / Axioma Vision — Backend POC de Detecção de Furtos em Mercados
(YOLOv8-Pose + Blur LGPD + Gravação Anotada + Juiz VLM + Webhook Situator)
"""

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI(title="Axioma Vision - Shoplifting POC API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pose_model = YOLO(os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt"))

RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)
app.mount("/recordings", StaticFiles(directory=str(RECORDINGS_DIR)), name="recordings")

VLM_API_URL = os.getenv("VLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
VLM_API_KEY = os.getenv("VLM_API_KEY", "")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen/qwen2.5-vl-72b-instruct")
SITUATOR_WEBHOOK_URL = os.getenv("SITUATOR_WEBHOOK_URL", "")
SITUATOR_TOKEN = os.getenv("SITUATOR_TOKEN", "")

SKELETON_LINKS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def _blur_face_lgpd(frame, kpts) -> None:
    """Aplica blur nos keypoints faciais (0=nariz, 1..4=olhos/orelhas) para conformidade LGPD."""
    face_pts = [pt for pt in kpts[:5] if float(pt[0]) > 0 and float(pt[1]) > 0]
    if not face_pts:
        return
    xs = [float(p[0]) for p in face_pts]
    ys = [float(p[1]) for p in face_pts]
    pad = 22
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(frame.shape[1], int(max(xs)) + pad)
    y2 = min(frame.shape[0], int(max(ys)) + pad)
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)


def _extract_pose_suspicion(video_path: str, out_name: str, max_frames: int = 35) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo enviado (formato inválido ou corrompido).")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Vídeo sem dimensões válidas.")

    step = max(1, int(fps // 5))  # amostra ~5 FPS
    rec_filename = f"annotated_{out_name}.webm"
    rec_path = RECORDINGS_DIR / rec_filename
    writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"VP80"), 5.0, (width, height))
    if not writer.isOpened():
        rec_filename = f"annotated_{out_name}.mp4"
        rec_path = RECORDINGS_DIR / rec_filename
        writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))

    frame_idx = 0
    analyzed = 0
    suspicious_frames = 0
    people_detected_frames = 0
    best_frame_b64 = None
    best_score = 0.0

    while cap.isOpened() and analyzed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        results = pose_model.track(frame, persist=True, verbose=False)
        frame_score = 0.0
        person_count = 0

        for res in results:
            if res.keypoints is None or res.keypoints.xy is None:
                continue
            boxes = res.boxes.xyxy if (res.boxes is not None and res.boxes.xyxy is not None) else None
            track_ids = res.boxes.id if (res.boxes is not None and res.boxes.id is not None) else None

            for idx, kpts in enumerate(res.keypoints.xy):
                if len(kpts) < 13:
                    continue
                person_count += 1
                _blur_face_lgpd(frame, kpts)

                # Desenha esqueleto
                for a, b in SKELETON_LINKS:
                    if a < len(kpts) and b < len(kpts):
                        xa, ya = int(kpts[a][0]), int(kpts[a][1])
                        xb, yb = int(kpts[b][0]), int(kpts[b][1])
                        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                            cv2.line(frame, (xa, ya), (xb, yb), (52, 211, 153), 2)

                l_wrist, r_wrist = kpts[9], kpts[10]
                l_hip, r_hip = kpts[11], kpts[12]
                l_sh = kpts[5]
                p_score = 0.0

                if float(l_hip[0]) > 0 and float(l_sh[0]) > 0:
                    torso_h = max(float(abs(l_hip[1] - l_sh[1])), 40.0)
                    dists = []
                    if float(l_wrist[0]) > 0:
                        dists.append(float(((l_wrist[0] - l_hip[0]) ** 2 + (l_wrist[1] - l_hip[1]) ** 2) ** 0.5))
                    if float(r_wrist[0]) > 0 and float(r_hip[0]) > 0:
                        dists.append(float(((r_wrist[0] - r_hip[0]) ** 2 + (r_wrist[1] - r_hip[1]) ** 2) ** 0.5))
                    if dists:
                        min_ratio = min(dists) / torso_h
                        if min_ratio < 0.42:
                            p_score = min(0.95, 1.0 - min_ratio)
                            frame_score = max(frame_score, p_score)

                if boxes is not None and idx < len(boxes):
                    x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
                    tid = int(track_ids[idx]) if (track_ids is not None and idx < len(track_ids)) else (idx + 1)
                    b_color = (0, 0, 255) if p_score >= 0.65 else (56, 189, 248)
                    label = f"ID #{tid} | {int(p_score * 100)}% {'SUSPEITA' if p_score >= 0.65 else 'NORMAL'}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, 2)
                    cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, b_color, 2)

        if person_count > 0:
            people_detected_frames += 1
        if frame_score >= 0.60:
            suspicious_frames += 1
        if frame_score > best_score:
            best_score = frame_score
            _, buf = cv2.imencode(".jpg", frame)
            best_frame_b64 = base64.b64encode(buf).decode("utf-8")

        hud_color = (0, 0, 255) if frame_score >= 0.65 else (56, 189, 248)
        cv2.putText(
            frame,
            f"AXIOMA LGPD-SAFE | Pessoas: {person_count} | Score: {frame_score:.2f}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            hud_color,
            2,
        )
        writer.write(frame)
        analyzed += 1
        frame_idx += 1

    cap.release()
    writer.release()

    if analyzed == 0:
        raise ValueError("Nenhum frame legível encontrado no arquivo de vídeo.")

    overall_score = round(min(0.98, best_score * 0.7 + min(0.3, suspicious_frames * 0.05)), 2)
    return {
        "frames_analyzed": analyzed,
        "people_detected_frames": people_detected_frames,
        "suspicious_frames": suspicious_frames,
        "pose_score": overall_score,
        "best_frame_b64": best_frame_b64,
        "recording_path": str(rec_path),
        "recording_filename": rec_filename,
    }


async def _ask_vlm_judge(frame_b64: str | None, corridor: str) -> dict[str, Any]:
    if not VLM_API_KEY or not frame_b64:
        return {
            "used_vlm": False,
            "verdict": "SIMULADO_SEM_VLM_KEY",
            "reason": "Gesto com mão próxima ao quadril/bolso detectado pelo YOLOv8-Pose (configure VLM_API_KEY para confirmação multimodal).",
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
    return {"status": "ok", "model": "yolov8-pose", "lgpd_blur": "enabled"}


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    account: str = Form("Mercado Piloto 01"),
    zone: str = Form("Corredor 03 - Bebidas"),
) -> dict[str, Any]:
    t0 = time.time()
    raw_bytes = await file.read()
    if len(raw_bytes) < 100:
        raise HTTPException(status_code=400, detail="Arquivo de vídeo vazio ou muito pequeno.")

    suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"
    clean_stem = Path(file.filename or "clip").stem.replace(" ", "_")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        pose_res = _extract_pose_suspicion(tmp_path, clean_stem)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    vlm_res = (
        await _ask_vlm_judge(pose_res["best_frame_b64"], zone)
        if pose_res["pose_score"] >= 0.65
        else {
            "used_vlm": False,
            "verdict": "DESCARTADO_POR_POSE",
            "reason": "Score de pose abaixo do limiar (0.65); economizou chamada ao VLM.",
        }
    )

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
    if SITUATOR_WEBHOOK_URL and pose_res["pose_score"] >= 0.65:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(SITUATOR_WEBHOOK_URL, json=situator_payload)
            webhook_sent = True

    return {
        "elapsed_ms": int((time.time() - t0) * 1000),
        "pose_analysis": {
            "frames_analyzed": pose_res["frames_analyzed"],
            "people_detected_frames": pose_res["people_detected_frames"],
            "suspicious_frames": pose_res["suspicious_frames"],
            "pose_score": pose_res["pose_score"],
            "recording_path": pose_res["recording_path"],
            "recording_filename": pose_res["recording_filename"],
        },
        "vlm_judge": vlm_res,
        "situator_webhook_sent": webhook_sent,
        "situator_payload": situator_payload,
    }
