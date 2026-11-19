"""
Seventh / Axioma Vision — Backend POC de Detecção de Furtos em Mercados
(YOLOv8-Pose.track + Storyboard Temporal VLM + Blur LGPD + Cache SHA-256
+ Integração Real D-Guard HTTP/CGI e Situator Webhook)
"""

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="Axioma Vision - Seventh POC API", version="0.4.0")

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
VLM_MODEL = os.getenv("VLM_MODEL", "google/gemini-2.5-flash-lite")
VLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "VLM_FALLBACK_MODELS",
        "qwen/qwen3-vl-30b-a3b-instruct,qwen/qwen3-vl-8b-instruct",
    ).split(",")
    if m.strip()
]
SITUATOR_WEBHOOK_URL = os.getenv("SITUATOR_WEBHOOK_URL", "http://127.0.0.1:8090/webhook")
SITUATOR_TOKEN = os.getenv("SITUATOR_TOKEN", "demo-token-situator")
DGUARD_URL = os.getenv("DGUARD_URL", "http://127.0.0.1:80")
DGUARD_USER = os.getenv("DGUARD_USER", "admin")
DGUARD_PASS = os.getenv("DGUARD_PASS", "")
DGUARD_CAMERA_ID = os.getenv("DGUARD_CAMERA_ID", "1")
DGUARD_CUSTOM_EVENT_GUID = os.getenv("DGUARD_CUSTOM_EVENT_GUID", "")
MAX_VLM_CALLS_PER_HOUR = int(os.getenv("MAX_VLM_CALLS_PER_HOUR", "60"))

# Cache em memória por SHA-256 do vídeo + parâmetros para evitar chamadas repetidas de VLM
ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
VLM_CALL_TIMESTAMPS: list[float] = []
LATEST_ANNOTATED_VIDEO: Path | None = None

SKELETON_LINKS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def _blur_face_lgpd(frame: np.ndarray, kpts: Any) -> None:
    """Aplica blur Gaussiano nos keypoints faciais (0..4) para conformidade LGPD."""
    face_pts = [pt for pt in kpts[:5] if float(pt[0]) > 0 and float(pt[1]) > 0]
    if not face_pts:
        return
    xs = [float(p[0]) for p in face_pts]
    ys = [float(p[1]) for p in face_pts]
    pad = 24
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(frame.shape[1], int(max(xs)) + pad)
    y2 = min(frame.shape[0], int(max(ys)) + pad)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (31, 31), 0)


def _angle_deg(a: Any, b: Any, c: Any) -> float:
    """Calcula o ângulo (em graus) no vértice b entre segmentos ba e bc."""
    bax, bay = float(a[0] - b[0]), float(a[1] - b[1])
    bcx, bcy = float(c[0] - b[0]), float(c[1] - b[1])
    na, nc = math.hypot(bax, bay), math.hypot(bcx, bcy)
    if na < 1e-3 or nc < 1e-3:
        return 180.0
    cos_val = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (na * nc)))
    return math.degrees(math.acos(cos_val))


def _kpt_ok(kpts: Any, confs: Any, idx: int, min_conf: float = 0.30) -> bool:
    """Verifica se o keypoint possui coordenadas válidas e confiança mínima."""
    if idx >= len(kpts) or float(kpts[idx][0]) <= 0 or float(kpts[idx][1]) <= 0:
        return False
    if confs is not None and idx < len(confs) and float(confs[idx]) < min_conf:
        return False
    return True


def _score_person_pose(kpts: Any, confs: Any = None) -> tuple[float, str]:
    """Avalia biomecânica de ocultação (flexão de cotovelo + mão junto ao tronco/bolso) sem confundir braços relaxados."""
    torso_hs = []
    if _kpt_ok(kpts, confs, 5) and _kpt_ok(kpts, confs, 11):
        torso_hs.append(abs(float(kpts[11][1]) - float(kpts[5][1])))
    if _kpt_ok(kpts, confs, 6) and _kpt_ok(kpts, confs, 12):
        torso_hs.append(abs(float(kpts[12][1]) - float(kpts[6][1])))
    if not torso_hs:
        return 0.0, "Tronco não visível"

    torso_h = max(max(torso_hs), 35.0)
    hip_xs = [float(kpts[i][0]) for i in (11, 12) if _kpt_ok(kpts, confs, i)]
    sh_xs = [float(kpts[i][0]) for i in (5, 6) if _kpt_ok(kpts, confs, i)]
    hip_cx = sum(hip_xs) / len(hip_xs)

    span_hip = abs(float(kpts[11][0]) - float(kpts[12][0])) if len(hip_xs) == 2 else 0.0
    span_sh = abs(float(kpts[5][0]) - float(kpts[6][0])) if len(sh_xs) == 2 else 0.0
    hip_w = max(span_hip, span_sh, torso_h * 0.42)

    sh_y = min(float(kpts[i][1]) for i in (5, 6) if _kpt_ok(kpts, confs, i))
    hp_y = max(float(kpts[i][1]) for i in (11, 12) if _kpt_ok(kpts, confs, i))

    best_score = 0.05
    best_reason = "Braços relaxados ao longo do corpo (postura normal)"

    for side_label, (sh_i, el_i, wr_i) in [("esq", (5, 7, 9)), ("dir", (6, 8, 10))]:
        if not (_kpt_ok(kpts, confs, sh_i) and _kpt_ok(kpts, confs, el_i) and _kpt_ok(kpts, confs, wr_i)):
            continue
        ang = _angle_deg(kpts[sh_i], kpts[el_i], kpts[wr_i])
        wx, wy = float(kpts[wr_i][0]), float(kpts[wr_i][1])
        dist_torso_cx = abs(wx - hip_cx) / hip_w
        in_jacket_zone = (sh_y + 0.08 * torso_h) <= wy <= (hp_y + 0.18 * torso_h)

        if ang < 130.0 and dist_torso_cx < 0.58 and in_jacket_zone:
            sc = 0.72 + min(0.22, (130.0 - ang) / 160.0)
            if sc > best_score:
                best_score = sc
                best_reason = f"Cotovelo flexionado ({ang:.0f}°) com pulso junto ao tronco/roupa ({side_label})"
        elif ang < 108.0 and (sh_y - 0.15 * torso_h) <= wy <= (hp_y + 0.25 * torso_h):
            sc = 0.66 + min(0.22, (108.0 - ang) / 140.0)
            if sc > best_score:
                best_score = sc
                best_reason = f"Flexão acentuada ({ang:.0f}°) com mão em zona de bolso/prateleira ({side_label})"
        elif ang < 135.0 and dist_torso_cx < 0.78 and in_jacket_zone:
            if 0.45 > best_score:
                best_score = 0.45
                best_reason = f"Movimento moderado próximo à cintura ({ang:.0f}°)"

    return round(min(0.95, best_score), 2), best_reason


def _build_storyboard_b64(clean_frames: list[np.ndarray], annotated_frames: list[np.ndarray], peak_pos: int) -> tuple[str, str]:
    """Gera colagem 2x2 cronológica (T1..T4) com quadros limpos (apenas blur LGPD, sem linhas cobrindo as mãos)."""
    n = len(clean_frames)
    if n == 0:
        return "", ""
    if n <= 4:
        idxs = [min(i, n - 1) for i in range(4)]
    else:
        idxs = [
            int(n * 0.12),
            int(n * 0.40),
            max(int(n * 0.58), min(int(n * 0.78), peak_pos)),
            int(n * 0.90),
        ]
        idxs = sorted(set(idxs))
        while len(idxs) < 4:
            for cand in [0, int(n * 0.30), int(n * 0.65), n - 1]:
                if cand not in idxs:
                    idxs.append(cand)
                if len(idxs) == 4:
                    break
            if len(idxs) < 4:
                idxs.append(min(n - 1, len(idxs)))
        idxs = sorted(idxs[:4])

    tiles = []
    for t_i, f_idx in enumerate(idxs):
        tile = cv2.resize(clean_frames[f_idx], (400, 300))
        cv2.putText(tile, f"T{t_i + 1}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        tiles.append(tile)

    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    _, buf_grid = cv2.imencode(".jpg", grid)
    ev_frame = annotated_frames[min(peak_pos, len(annotated_frames) - 1)]
    _, buf_peak = cv2.imencode(".jpg", ev_frame)
    return base64.b64encode(buf_grid).decode("utf-8"), base64.b64encode(buf_peak).decode("utf-8")


def _extract_pose_suspicion(video_path: str, out_stem: str, vid_hash_short: str) -> dict[str, Any]:
    """Processa o vídeo COMPLETO do início ao fim preservando duração original, skeletons e blur LGPD."""
    global LATEST_ANNOTATED_VIDEO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Não foi possível abrir o vídeo enviado (formato inválido ou corrompido).")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    if fps <= 0:
        fps = 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = round(total_frames / fps, 1) if total_frames > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Vídeo sem dimensões válidas.")

    if width > 576:
        scale = 576.0 / width
        width = 576
        height = max(2, int(round(height * scale)) // 2 * 2)

    # Reseta estado de rastreamento entre vídeos distintos para manter IDs limpos
    if getattr(pose_model, "predictor", None) is not None and getattr(pose_model.predictor, "trackers", None):
        pose_model.predictor.trackers.clear()

    if total_frames > 0:
        target_samples = min(total_frames, max(24, min(50, int(round(max(duration_s, 1.0) * 2.5)))))
        step = max(1, total_frames // target_samples)
        expected_written = (total_frames + step - 1) // step
        out_fps = max(1.0, expected_written / duration_s) if duration_s > 0 else 4.0
    else:
        step = max(1, int(round(fps / 3.0)))
        out_fps = 3.0

    rec_filename = f"annotated_{out_stem}_{vid_hash_short}.webm"
    rec_path = RECORDINGS_DIR / rec_filename
    writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"VP80"), out_fps, (width, height))
    if not writer.isOpened():
        rec_filename = f"annotated_{out_stem}_{vid_hash_short}.mp4"
        rec_path = RECORDINGS_DIR / rec_filename
        writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))

    frame_idx = 0
    analyzed = 0
    suspicious_frames = 0
    people_detected_frames = 0
    best_score = 0.0
    best_pose_reason = "Nenhuma pessoa suspeita detectada"
    clean_lgpd_frames: list[np.ndarray] = []
    sampled_annotated: list[np.ndarray] = []
    peak_sampled_idx = 0

    try:
        while cap.isOpened():
            if frame_idx % step != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            results = pose_model.track(frame, imgsz=384, persist=True, verbose=False)
            frame_score = 0.0
            person_count = 0

            # Primeiro aplica apenas blur LGPD e guarda cópia limpa para o Juiz VLM
            for res in results:
                if res.keypoints is not None and res.keypoints.xy is not None:
                    for kpts in res.keypoints.xy:
                        if len(kpts) >= 13:
                            _blur_face_lgpd(frame, kpts)
            clean_copy = cv2.resize(frame, (480, max(2, int(480 * height / width))))

            for res in results:
                if res.keypoints is None or res.keypoints.xy is None:
                    continue
                boxes = res.boxes.xyxy if (res.boxes is not None and res.boxes.xyxy is not None) else None
                track_ids = res.boxes.id if (res.boxes is not None and res.boxes.id is not None) else None
                kpt_confs = res.keypoints.conf if (res.keypoints is not None and res.keypoints.conf is not None) else None

                for idx, kpts in enumerate(res.keypoints.xy):
                    if len(kpts) < 13:
                        continue
                    person_count += 1
                    confs_i = kpt_confs[idx] if (kpt_confs is not None and idx < len(kpt_confs)) else None

                    for a, b in SKELETON_LINKS:
                        if a < len(kpts) and b < len(kpts):
                            xa, ya = int(kpts[a][0]), int(kpts[a][1])
                            xb, yb = int(kpts[b][0]), int(kpts[b][1])
                            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                                cv2.line(frame, (xa, ya), (xb, yb), (52, 211, 153), 2)

                    p_score, p_reason = _score_person_pose(kpts, confs_i)
                    frame_score = max(frame_score, p_score)
                    if p_score >= best_score:
                        best_pose_reason = p_reason

                    if boxes is not None and idx < len(boxes):
                        x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
                        tid = int(track_ids[idx]) if (track_ids is not None and idx < len(track_ids)) else (idx + 1)
                        b_color = (0, 0, 255) if p_score >= 0.65 else (56, 189, 248)
                        label = f"ID #{tid} | {int(p_score * 100)}% {'SUSPEITA' if p_score >= 0.65 else 'NORMAL'}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), b_color, 2)
                        cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, b_color, 2)

            if person_count > 0:
                people_detected_frames += 1
            if frame_score >= 0.65:
                suspicious_frames += 1

            hud_color = (0, 0, 255) if frame_score >= 0.65 else (56, 189, 248)
            cv2.putText(
                frame,
                f"AXIOMA LGPD-SAFE | Pessoas: {person_count} | Score: {frame_score:.2f}",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                hud_color,
                2,
            )
            writer.write(frame)

            clean_lgpd_frames.append(clean_copy)
            sampled_annotated.append(cv2.resize(frame, (480, max(2, int(480 * height / width)))))
            if frame_score > best_score:
                best_score = frame_score
                peak_sampled_idx = len(clean_lgpd_frames) - 1

            analyzed += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if analyzed == 0:
        raise ValueError("Nenhum frame legível encontrado no arquivo de vídeo.")

    storyboard_b64, peak_b64 = _build_storyboard_b64(clean_lgpd_frames, sampled_annotated, peak_sampled_idx)
    ev_filename = f"evidence_{out_stem}_{vid_hash_short}.jpg"
    if peak_b64:
        (RECORDINGS_DIR / ev_filename).write_bytes(base64.b64decode(peak_b64))
    else:
        ev_filename = ""

    LATEST_ANNOTATED_VIDEO = rec_path
    overall_score = round(min(0.98, best_score * 0.72 + min(0.28, suspicious_frames * 0.03)), 2)
    return {
        "duration_s": duration_s if duration_s > 0 else round(analyzed / out_fps, 1),
        "frames_analyzed": analyzed,
        "people_detected_frames": people_detected_frames,
        "suspicious_frames": suspicious_frames,
        "pose_score": overall_score,
        "pose_reason": best_pose_reason,
        "storyboard_b64": storyboard_b64,
        "recording_path": str(rec_path),
        "recording_filename": rec_filename,
        "evidence_filename": ev_filename,
    }


def _check_vlm_rate_limit() -> bool:
    """Impede estouro de custo limitando chamadas VLM por janela móvel de 1 hora."""
    now = time.time()
    VLM_CALL_TIMESTAMPS[:] = [ts for ts in VLM_CALL_TIMESTAMPS if now - ts < 3600]
    if len(VLM_CALL_TIMESTAMPS) >= MAX_VLM_CALLS_PER_HOUR:
        return False
    VLM_CALL_TIMESTAMPS.append(now)
    return True


async def _ask_vlm_judge(storyboard_b64: str, zone: str) -> dict[str, Any]:
    """Consulta o Juiz VLM com storyboard 2x2 (já com blur LGPD) e fallback automático contra HTTP 429."""
    if not VLM_API_KEY:
        return {
            "used_vlm": False,
            "confirmed_theft": False,
            "verdict": "VLM_NAO_CONFIGURADO",
            "reason": "Configure VLM_API_KEY no arquivo poc-backend/.env.",
        }
    if not storyboard_b64:
        return {"used_vlm": False, "confirmed_theft": False, "verdict": "SEM_FRAME", "reason": "Frame indisponível."}
    if not _check_vlm_rate_limit():
        return {
            "used_vlm": False,
            "confirmed_theft": False,
            "verdict": "GUARDRAIL_RATE_LIMIT",
            "reason": f"Limite horário de segurança ({MAX_VLM_CALLS_PER_HOUR} chamadas VLM/h) atingido.",
        }

    prompt = (
        f"Você é o Juiz VLM antifurto da Axioma Vision monitorando a zona '{zone}'. "
        "A imagem mostra um storyboard temporal 2x2 em ordem cronológica (T1 -> T2 -> T3 -> T4) de uma câmera CFTV "
        "(rostos desfocados por LGPD). Avalie se há FURTO/OCULTAÇÃO real de mercadoria "
        "(ex: cliente pegando produto da gôndola/balcão e escondendo dentro do casaco, roupa, bolso ou mochila para sair sem pagar) "
        "ou apenas comportamento normal (olhar prateleira, pegar item sem esconder, mexer no celular, conversar). "
        'Responda APENAS JSON válido: {"suspeita": true ou false, "motivo": "explicação curta e objetiva em português"}.'
    )

    candidate_models = [VLM_MODEL] + [m for m in VLM_FALLBACK_MODELS if m != VLM_MODEL]
    headers = {"Authorization": f"Bearer {VLM_API_KEY}", "Content-Type": "application/json"}
    last_err = ""
    raw_text = ""
    used_model = VLM_MODEL

    async with httpx.AsyncClient(timeout=28.0) as client:
        for model_id in candidate_models:
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{storyboard_b64}"}},
                        ],
                    }
                ],
                "max_tokens": 140,
                "temperature": 0.1,
            }
            try:
                resp = await client.post(VLM_API_URL, json=payload, headers=headers)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = f"{model_id}: HTTP {resp.status_code}"
                    continue
                resp.raise_for_status()
                raw_text = resp.json()["choices"][0]["message"]["content"].strip()
                used_model = model_id
                break
            except Exception as exc:
                last_err = f"{model_id}: {exc}"
                continue

    if not raw_text:
        return {
            "used_vlm": False,
            "confirmed_theft": False,
            "verdict": "ERRO_VLM",
            "reason": f"Falha nos modelos VLM ({last_err}).",
        }

    confirmed = False
    reason = raw_text
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            confirmed = bool(parsed.get("suspeita", False))
            reason = str(parsed.get("motivo", raw_text))
        except json.JSONDecodeError:
            confirmed = '"suspeita": true' in raw_text.lower() or "suspeita: true" in raw_text.lower()

    return {
        "used_vlm": True,
        "model": used_model,
        "confirmed_theft": confirmed,
        "verdict": "CONFIRMADO_VLM" if confirmed else "DESCARTADO_PELO_VLM",
        "reason": reason,
    }


def _dguard_auth(user: str, password: str) -> tuple[str, str] | None:
    return (user, password) if user else None


async def _probe_dguard(base_url: str, user: str, password: str) -> dict[str, Any]:
    """Consulta D-Guard HTTP/CGI (conforme Interface CGI V4.pdf) e descobre câmeras/eventos customizados."""
    urls_to_try = [base_url.rstrip("/")]
    if base_url.rstrip("/") == "http://127.0.0.1:80":
        urls_to_try.append("http://127.0.0.1:8081")

    auth = _dguard_auth(user, password)
    async with httpx.AsyncClient(timeout=0.5) as client:
        for url in urls_to_try:
            try:
                srv = await client.get(f"{url}/servidor.cgi", auth=auth)
                if srv.status_code < 500:
                    cams_resp = await client.get(f"{url}/camerasnomes.cgi", auth=auth)
                    ev_resp = await client.get(f"{url}/customevents/getcustomevents.cgi", auth=auth)
                    custom_events: list[dict[str, Any]] = []
                    if ev_resp.status_code == 200 and ev_resp.text.strip().startswith("["):
                        try:
                            custom_events = ev_resp.json()
                        except Exception:
                            custom_events = []
                    return {
                        "reachable": True,
                        "url": url,
                        "server": srv.text.strip(),
                        "status_code": srv.status_code,
                        "cameras_raw": cams_resp.text.strip()[:300],
                        "custom_events": custom_events,
                    }
            except Exception:
                continue

    return {
        "reachable": False,
        "url": base_url,
        "server": "Offline nesta máquina (abra o aplicativo D-Guard Center no Windows para ativar a API CGI na porta 80/8081).",
        "cameras_raw": "",
        "custom_events": [],
    }


async def _dispatch_to_dguard(
    base_url: str,
    user: str,
    password: str,
    camera_id: str,
    custom_guid: str,
    osd_text: str,
) -> dict[str, Any]:
    """Executa comandos CGI reais no D-Guard em caso de alerta confirmado (OSD, Movimento, Gravação e Evento)."""
    probe = await _probe_dguard(base_url, user, password)
    if not probe["reachable"]:
        return {
            "executed": False,
            "reachable": False,
            "target_url": base_url,
            "message": probe["server"],
            "actions": [],
        }

    url = probe["url"]
    auth = _dguard_auth(user, password)
    guid = custom_guid.strip()
    if not guid and probe["custom_events"]:
        guid = str(probe["custom_events"][0].get("guid", "")).strip()

    endpoints = [
        f"{url}/setcameraosd.cgi?camera={camera_id}&osd={quote_plus(osd_text)}&time=15",
        f"{url}/camerasetmotion.cgi?camera={camera_id}&estado=on",
        f"{url}/camerasetgravacao.cgi?camera={camera_id}&estado=on",
    ]
    if guid:
        endpoints.append(f"{url}/customevents/triggercustomevent.cgi?guid={quote_plus(guid)}")

    actions: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=3.0) as client:
        for ep in endpoints:
            try:
                r = await client.get(ep, auth=auth)
                actions.append({"endpoint": ep.replace(url, ""), "status": r.status_code, "response": r.text.strip()[:120]})
            except Exception as exc:
                actions.append({"endpoint": ep.replace(url, ""), "status": 0, "response": str(exc)})

    return {
        "executed": True,
        "reachable": True,
        "target_url": url,
        "server": probe["server"],
        "actions": actions,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "yolov8-pose",
        "vlm_model": VLM_MODEL if VLM_API_KEY else "not_configured",
        "lgpd_blur": "enabled",
        "cache_entries": len(ANALYSIS_CACHE),
    }


@app.get("/seventh/status")
async def seventh_status(
    dguard_url: str = Query(default=DGUARD_URL),
    dguard_user: str = Query(default=DGUARD_USER),
    dguard_pass: str = Query(default=DGUARD_PASS),
    situator_url: str = Query(default=SITUATOR_WEBHOOK_URL),
) -> dict[str, Any]:
    """Verifica conectividade real ao vivo com o D-Guard CGI e o receptor Situator."""
    dg = await _probe_dguard(dguard_url, dguard_user, dguard_pass)
    sit_ok = False
    sit_info: Any = "Offline"
    if situator_url:
        events_ep = re.sub(r"/webhook/?$", "/events", situator_url.strip())
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                sr = await client.get(events_ep)
                sit_ok = sr.status_code < 500
                sit_info = sr.json() if sr.headers.get("content-type", "").startswith("application/json") else sr.text[:120]
        except Exception as exc:
            sit_info = str(exc)

    return {
        "dguard": dg,
        "situator": {"reachable": sit_ok, "webhook_url": situator_url, "info": sit_info},
    }


@app.post("/seventh/trigger-test")
async def trigger_seventh_test(
    account: str = Form("Mercado Piloto 01"),
    zone: str = Form("Corredor 03 - Bebidas"),
    situator_url: str = Form(SITUATOR_WEBHOOK_URL),
    dguard_url: str = Form(DGUARD_URL),
    dguard_user: str = Form(DGUARD_USER),
    dguard_pass: str = Form(DGUARD_PASS),
    dguard_camera: str = Form(DGUARD_CAMERA_ID),
) -> dict[str, Any]:
    """Dispara evento real de teste para a fila do Situator e comandos CGI no D-Guard."""
    payload = {
        "token": SITUATOR_TOKEN,
        "eventCode": "AI_SHOPLIFTING_SUSPECT",
        "priority": "ALTA",
        "account": account,
        "zone": zone,
        "confidence": 0.91,
        "description": "[TESTE DE INTEGRAÇÃO REAL] Disparo validado para fila do Situator e OSD no D-Guard.",
        "recommendedAction": "Verificar câmera no D-Guard e validar abordagem preventiva no Situator.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sit_status: dict[str, Any] = {"sent": False, "url": situator_url}
    if situator_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(situator_url, json=payload)
                sit_status = {"sent": r.status_code < 400, "url": situator_url, "status": r.status_code, "body": r.text[:160]}
        except Exception as exc:
            sit_status = {"sent": False, "url": situator_url, "error": str(exc)}

    dg_res = await _dispatch_to_dguard(
        dguard_url, dguard_user, dguard_pass, dguard_camera, DGUARD_CUSTOM_EVENT_GUID,
        f"AXIOMA TESTE OK | {zone}",
    )
    return {"situator": sit_status, "dguard": dg_res, "payload": payload}


@app.get("/dguard/virtual-cam.mjpeg")
async def virtual_cam_mjpeg():
    """Stream MJPEG ao vivo do último vídeo analisado (permite cadastrar a IA como câmera IP dentro do D-Guard)."""
    target = LATEST_ANNOTATED_VIDEO
    if target is None or not target.exists():
        webms = sorted(RECORDINGS_DIR.glob("annotated_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        target = webms[0] if webms else None
    if target is None or not target.exists():
        raise HTTPException(status_code=404, detail="Nenhum vídeo anotado disponível ainda.")

    async def frame_gen():
        cap = cv2.VideoCapture(str(target))
        try:
            while True:
                ret, fr = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                _, jpg = cv2.imencode(".jpg", fr)
                data = jpg.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                await asyncio.sleep(0.18)
        finally:
            cap.release()

    return StreamingResponse(frame_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    account: str = Form("Mercado Piloto 01"),
    zone: str = Form("Corredor 03 - Bebidas"),
    situator_url: str = Form(SITUATOR_WEBHOOK_URL),
    dguard_url: str = Form(DGUARD_URL),
    dguard_user: str = Form(DGUARD_USER),
    dguard_pass: str = Form(DGUARD_PASS),
    dguard_camera: str = Form(DGUARD_CAMERA_ID),
) -> dict[str, Any]:
    t0 = time.time()
    raw_bytes = await file.read()
    if len(raw_bytes) < 100:
        raise HTTPException(status_code=400, detail="Arquivo de vídeo vazio ou muito pequeno.")

    orig_filename = file.filename or "clip.mp4"
    video_hash = hashlib.sha256(raw_bytes).hexdigest()
    short_hash = video_hash[:10]
    cache_key = f"{video_hash}:{account}:{zone}"
    if cache_key in ANALYSIS_CACHE:
        cached_resp = dict(ANALYSIS_CACHE[cache_key])
        cached_resp["cached"] = True
        cached_resp["elapsed_ms"] = int((time.time() - t0) * 1000)
        return cached_resp

    suffix = os.path.splitext(orig_filename)[1] or ".mp4"
    clean_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(orig_filename).stem)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        pose_res = await asyncio.to_thread(_extract_pose_suspicion, tmp_path, clean_stem, short_hash)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Guardrail de custo: só consulta VLM se pose_score >= 0.65 E houver >= 2 frames suspeitos
    if pose_res["pose_score"] >= 0.65 and pose_res["suspicious_frames"] >= 2:
        vlm_res = await _ask_vlm_judge(pose_res["storyboard_b64"], zone)
    else:
        vlm_res = {
            "used_vlm": False,
            "confirmed_theft": False,
            "verdict": "DESCARTADO_POR_POSE",
            "reason": f"Sem gesto crítico sustentado ({pose_res['pose_reason']}; {pose_res['suspicious_frames']} frames suspeitos). VLM não acionado (custo zero).",
        }

    is_real_alert = bool(vlm_res.get("confirmed_theft", False))
    situator_payload = {
        "token": SITUATOR_TOKEN,
        "eventCode": "AI_SHOPLIFTING_SUSPECT" if is_real_alert else "AI_FALSE_POSITIVE_FILTERED",
        "priority": "ALTA" if is_real_alert else "BAIXA",
        "account": account,
        "zone": zone,
        "videoFile": orig_filename,
        "durationSeconds": pose_res["duration_s"],
        "confidence": pose_res["pose_score"],
        "evidenceFile": pose_res["evidence_filename"],
        "description": vlm_res["reason"],
        "recommendedAction": (
            "Abordagem preventiva de atendimento ao cliente ('Posso ajudar?')"
            if is_real_alert
            else "Nenhuma ação necessária (descartado pela verificação de 2 estágios)"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    webhook_sent = False
    webhook_http_status: int | str = "Não enviado (sem alerta real)"
    if situator_url and is_real_alert:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                wr = await client.post(situator_url, json=situator_payload)
                webhook_sent = wr.status_code < 400
                webhook_http_status = f"HTTP {wr.status_code}"
        except Exception as exc:
            webhook_http_status = f"Erro: {exc}"

    dguard_res: dict[str, Any] = {"executed": False, "reachable": False, "actions": [], "message": "Sem alerta real para acionar D-Guard."}
    if is_real_alert:
        dguard_res = await _dispatch_to_dguard(
            dguard_url,
            dguard_user,
            dguard_pass,
            dguard_camera,
            DGUARD_CUSTOM_EVENT_GUID,
            f"ALERTA FURTO IA ({int(pose_res['pose_score']*100)}%) | {zone}",
        )

    response = {
        "cached": False,
        "filename": orig_filename,
        "video_sha256": video_hash[:12],
        "elapsed_ms": int((time.time() - t0) * 1000),
        "pose_analysis": {
            "duration_s": pose_res["duration_s"],
            "frames_analyzed": pose_res["frames_analyzed"],
            "people_detected_frames": pose_res["people_detected_frames"],
            "suspicious_frames": pose_res["suspicious_frames"],
            "pose_score": pose_res["pose_score"],
            "pose_reason": pose_res["pose_reason"],
            "recording_path": pose_res["recording_path"],
            "recording_filename": pose_res["recording_filename"],
            "evidence_filename": pose_res["evidence_filename"],
        },
        "vlm_judge": vlm_res,
        "situator_webhook_sent": webhook_sent,
        "situator_http_status": webhook_http_status,
        "dguard_integration": dguard_res,
        "situator_payload": situator_payload,
    }
    ANALYSIS_CACHE[cache_key] = response
    return response
