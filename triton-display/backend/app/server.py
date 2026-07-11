"""
FastAPI proxy for Triton inference visualization.

Endpoints:
  GET  /api/models          -> list READY ensemble models
  POST /api/infer/{model}   -> run inference on uploaded image
  GET  /health              -> health check
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
import tritonclient.http as httpclient
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

TRITON_HTTP_URL = os.environ.get("TRITON_HTTP_URL", "localhost:8000")
STATIC_DIR = os.environ.get("STATIC_DIR", "dist")


def get_triton_client():
    return httpclient.InferenceServerClient(url=TRITON_HTTP_URL, verbose=False)


app = FastAPI(title="Triton Display")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_labels_from_triton(model_name: str):
    """Fetch labels from Triton 'labels' Python backend model."""
    client = get_triton_client()
    model_name_arr = np.array([model_name.encode("utf-8")], dtype=object)
    input_tensor = httpclient.InferInput("model_name", [1], "BYTES")
    input_tensor.set_data_from_numpy(model_name_arr)
    output = httpclient.InferRequestedOutput("labels")
    response = client.infer("labels", inputs=[input_tensor], outputs=[output])
    labels = response.as_numpy("labels")
    return [l.decode("utf-8") for l in labels]


@app.get("/api/labels/{model_name}")
async def get_labels(model_name: str):
    """Return labels for a specific model."""
    try:
        labels = fetch_labels_from_triton(model_name)
        return {"model": model_name, "labels": labels}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No labels for model {model_name}: {e}")


@app.get("/api/labels")
async def list_all_labels():
    """Return labels for all READY ensemble models."""
    try:
        client = get_triton_client()
        models = client.get_model_repository_index()
        ensembles = [m["name"] for m in models if m["name"].endswith("_ensemble") and m.get("state") == "READY"]
        result = {}
        for name in ensembles:
            try:
                result[name] = fetch_labels_from_triton(name)
            except Exception:
                result[name] = []
        return {"labels": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Triton error: {e}")


@app.get("/api/models")
async def list_models():
    """Return READY ensemble models."""
    try:
        client = get_triton_client()
        models = client.get_model_repository_index()
        ensembles = [
            {
                "name": m["name"],
                "platform": m.get("platform", ""),
                "ready": m.get("state", "") == "READY",
            }
            for m in models
            if m["name"].endswith("_ensemble")
        ]
        return {"models": ensembles}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Triton error: {e}")


@app.get("/api/models/{model_name}/labels")
async def get_model_labels(model_name: str):
    """Convenience endpoint: labels for a specific model."""
    return await get_labels(model_name)


@app.post("/api/infer/{model_name}")
async def infer(
    model_name: str,
    image: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
):
    """Run inference on uploaded image and return parsed results."""
    try:
        contents = await image.read()

        # Decode image from buffer
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Add batch dimension: [1, H, W, 3]
        batched = np.expand_dims(img, 0)

        input_tensor = httpclient.InferInput("raw_image", batched.shape, "UINT8")
        input_tensor.set_data_from_numpy(batched)

        output_names = [
            "num_dets",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
            "transform_metadata",
        ]

        # Pose models also have keypoints
        if "pose" in model_name:
            output_names.append("detection_keypoints")

        outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

        client = get_triton_client()
        response = client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=outputs,
        )

        num_dets = response.as_numpy("num_dets")[0, 0].item()
        boxes = response.as_numpy("detection_boxes")[0]  # [max_dets, ...]
        scores = response.as_numpy("detection_scores")[0]
        classes = response.as_numpy("detection_classes")[0]
        transform = response.as_numpy("transform_metadata")[0].reshape(2, 3)

        result = {
            "image_shape": list(img.shape),
            "num_dets": int(num_dets),
            "transform": transform.tolist(),
            "detections": [],
        }

        for i in range(int(num_dets)):
            if scores[i] < conf_threshold:
                continue

            det = {
                "score": float(scores[i]),
                "class_id": int(classes[i]),
                "box": boxes[i].tolist(),
            }

            if "pose" in model_name:
                kpts = response.as_numpy("detection_keypoints")[0, i]
                det["keypoints"] = kpts.tolist()

            result["detections"].append(det)

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve built Vue frontend
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    if os.path.isdir(STATIC_DIR):
        index = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index):
            return FileResponse(index)
    raise HTTPException(status_code=404, detail="Not found")
