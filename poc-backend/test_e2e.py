"""
Teste E2E rápido da POC sem precisar de câmera física:
Gera um vídeo MP4 sintético de 3s, envia para POST /analyze e valida a resposta JSON.
"""

import os
import tempfile

import cv2
import httpx
import numpy as np

API_URL = os.getenv("POC_API_URL", "http://127.0.0.1:8000/analyze")


def create_dummy_mp4(path: str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 15.0, (640, 360))
    for i in range(45):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Seventh POC Frame {i}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (56, 189, 248), 2)
        out.write(frame)
    out.release()


def main() -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        mp4_path = tmp.name
    try:
        create_dummy_mp4(mp4_path)
        with open(mp4_path, "rb") as f:
            resp = httpx.post(
                API_URL,
                files={"file": ("teste_corredor.mp4", f, "video/mp4")},
                data={"account": "Mercado Piloto 01", "zone": "Corredor 03 - Bebidas"},
                timeout=60.0,
            )
        resp.raise_for_status()
        print("OK! Resposta da API POC:")
        print(resp.json())
    finally:
        if os.path.exists(mp4_path):
            os.remove(mp4_path)


if __name__ == "__main__":
    main()
