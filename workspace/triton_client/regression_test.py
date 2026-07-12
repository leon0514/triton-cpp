import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import os

img_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'bus.jpg')
img = Image.open(img_path).convert('RGB')
arr = np.array(img, dtype=np.uint8)
arr = np.expand_dims(arr, axis=0)

client = httpclient.InferenceServerClient(url='localhost:48000')

for model in ['yolo11_ensemble', 'yolo11_obb_ensemble', 'yolo11_pose_ensemble', 'yolo11_seg_ensemble']:
    inputs = [httpclient.InferInput('raw_image', arr.shape, 'UINT8')]
    inputs[0].set_data_from_numpy(arr)
    outputs = [
        httpclient.InferRequestedOutput('num_dets'),
        httpclient.InferRequestedOutput('detection_boxes'),
        httpclient.InferRequestedOutput('detection_scores'),
        httpclient.InferRequestedOutput('detection_classes'),
    ]
    if 'seg' in model:
        outputs.extend([
            httpclient.InferRequestedOutput('detection_masks'),
            httpclient.InferRequestedOutput('mask_offsets'),
            httpclient.InferRequestedOutput('mask_shapes'),
        ])
    if 'pose' in model:
        outputs.append(httpclient.InferRequestedOutput('detection_keypoints'))
    response = client.infer(model, inputs=inputs, outputs=outputs)
    num_dets = response.as_numpy('num_dets')[0, 0]
    print(f"{model}: num_dets={num_dets}")
    scores = response.as_numpy('detection_scores')[0]
    print(f"  top scores: {scores[:num_dets].tolist()}")
print("Regression OK")
