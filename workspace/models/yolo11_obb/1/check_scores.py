import onnxruntime as ort
import numpy as np
import cv2


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = cv2.resize(img, (nw, nh))
    return padded


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


img = cv2.imread("../../../images/bus.jpg")
img = letterbox(img)
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)[None]

sess = ort.InferenceSession("yolo11n-obb.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"images": img})[0][0]  # [20, 8400]
cls = out[5:]
scores = sigmoid(cls).max(axis=0)

for thresh in [0.1, 0.25, 0.5, 0.7, 0.8]:
    count = (scores > thresh).sum()
    print(f"thresh={thresh}: {count} anchors")
print(f"max score: {scores.max():.4f}")
print(f"top 10 scores: {np.sort(scores)[::-1][:10]}")
