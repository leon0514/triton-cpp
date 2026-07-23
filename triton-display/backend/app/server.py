"""
FastAPI proxy for Triton inference visualization.

Endpoints:
  GET  /api/models                -> list READY ensemble models
  GET  /api/models/all            -> list ALL models with ready state
  GET  /api/models/{model}/config -> get model config
  POST /api/models/{model}/load   -> load a model
  POST /api/models/{model}/unload -> unload a model
  POST /api/infer/{model}         -> run inference on uploaded image
  GET  /health                    -> health check
"""

import os
import base64
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
    response = client.infer("CUSTOM_LABELS", inputs=[input_tensor], outputs=[output])
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
        ensembles = [m["name"] for m in models if m["name"].endswith("_ENSEMBLE") and m.get("state") == "READY"]
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
            if m["name"].endswith("_ENSEMBLE")
        ]
        return {"models": ensembles}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Triton error: {e}")


@app.get("/api/models/{model_name}/labels")
async def get_model_labels(model_name: str):
    """Convenience endpoint: labels for a specific model."""
    return await get_labels(model_name)


@app.get("/api/models/all")
async def list_all_models():
    """Return all models in the repository with their ready state."""
    try:
        client = get_triton_client()
        models = client.get_model_repository_index()
        return {
            "models": [
                {
                    "name": m["name"],
                    "platform": m.get("platform", ""),
                    "state": m.get("state", ""),
                    "ready": m.get("state", "") == "READY",
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Triton error: {e}")


@app.get("/api/models/{model_name}/config")
async def get_model_config(model_name: str):
    """Return Triton model configuration (config.pbtxt parsed)."""
    try:
        client = get_triton_client()
        config = client.get_model_config(model_name)
        return {"name": model_name, "config": config}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Config for {model_name} not found: {e}")


@app.post("/api/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a model into Triton."""
    try:
        client = get_triton_client()
        client.load_model(model_name)
        return {"name": model_name, "action": "load", "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load {model_name}: {e}")


@app.post("/api/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Unload a model from Triton."""
    try:
        client = get_triton_client()
        client.unload_model(model_name)
        return {"name": model_name, "action": "unload", "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to unload {model_name}: {e}")


@app.post("/api/infer/{model_name}")
async def infer(
    model_name: str,
    image: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    raw_mask: bool = Form(False),
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

        client = get_triton_client()

        model_name_lower = model_name.lower()

        # Classification ensemble returns top-k classes and scores
        if "classifier" in model_name_lower:
            output_names = ["classes", "scores"]
            outputs = [httpclient.InferRequestedOutput(name) for name in output_names]
            response = client.infer(
                model_name=model_name,
                inputs=[input_tensor],
                outputs=outputs,
            )

            classes_arr = response.as_numpy("classes")[0]  # [top_k]
            scores_arr = response.as_numpy("scores")[0]

            return {
                "model_type": "classifier",
                "image_shape": list(img.shape),
                "classes": classes_arr.tolist(),
                "scores": scores_arr.tolist(),
                "top_k": int(classes_arr.shape[0]),
                "detections": [],
            }

        output_names = [
            "num_dets",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ]

        # Pose models also have keypoints
        if "pose" in model_name_lower:
            output_names.append("detection_keypoints")

        # Segmentation models have mask outputs
        if "seg" in model_name_lower:
            output_names.extend([
                "detection_masks",
                "mask_offsets",
                "mask_shapes",
            ])

        outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

        response = client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=outputs,
        )

        # reshape(-1, ...) 兼容有/无 batch 维度（SAHI max_batch_size=0 vs 直调 max_batch_size=16）
        num_dets = response.as_numpy("num_dets").reshape(-1)[0].item()
        boxes = response.as_numpy("detection_boxes").reshape(-1, response.as_numpy("detection_boxes").shape[-1])
        scores = response.as_numpy("detection_scores").reshape(-1)
        classes = response.as_numpy("detection_classes").reshape(-1)

        result = {
            "model_type": "detection",
            "image_shape": list(img.shape),
            "num_dets": int(num_dets),
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

            if "pose" in model_name_lower:
                kpts_raw = response.as_numpy("detection_keypoints")
                kpts = kpts_raw.reshape(-1, kpts_raw.shape[-2], kpts_raw.shape[-1])[i]
                det["keypoints"] = kpts.tolist()

            if "seg" in model_name_lower:
                masks_raw = response.as_numpy("detection_masks")
                masks_2d = masks_raw.reshape(-1, masks_raw.shape[-1])
                mask_slot = masks_2d.shape[-1]
                h, w = int(mask_slot ** 0.5), int(mask_slot ** 0.5)  # 默认正方形
                # 如果有 mask_shapes，读取实际尺寸
                try:
                    shapes_raw = response.as_numpy("mask_shapes")
                    shapes = shapes_raw.reshape(-1, shapes_raw.shape[-1])
                    h, w = int(shapes[i, 0]), int(shapes[i, 1])
                except Exception:
                    pass
                if mask_slot > 0:
                    det["mask_shape"] = [h, w]
                    mask_data = masks_2d[i].astype(np.float32)
                    if raw_mask:
                        det["mask"] = mask_data.tolist()
                    else:
                        det["mask"] = base64.b64encode(mask_data.tobytes()).decode('ascii')
                    det["mask_stats"] = {
                        "min": float(mask_data.min()),
                        "max": float(mask_data.max()),
                        "mean": float(mask_data.mean()),
                    }

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
