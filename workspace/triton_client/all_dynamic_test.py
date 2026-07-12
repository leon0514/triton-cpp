import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import os

img_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'bus.jpg')
img = Image.open(img_path).convert('RGB')
arr = np.array(img, dtype=np.uint8)
arr = np.expand_dims(arr, axis=0)

client = httpclient.InferenceServerClient(url='localhost:48000')

models = {
    'yolo11_ensemble': ['num_dets', 'detection_boxes', 'detection_scores', 'detection_classes'],
    'yolo11_obb_ensemble': ['num_dets', 'detection_boxes', 'detection_scores', 'detection_classes'],
    'yolo11_pose_ensemble': ['num_dets', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_keypoints'],
    'yolo11_seg_ensemble': ['num_dets', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks', 'mask_offsets', 'mask_shapes'],
    'yolov5_ensemble': ['num_dets', 'detection_boxes', 'detection_scores', 'detection_classes'],
    'yolo26_ensemble': ['num_dets', 'detection_boxes', 'detection_scores', 'detection_classes'],
    'rfdetr_ensemble': ['num_dets', 'detection_boxes', 'detection_scores', 'detection_classes'],
}

for model_name, output_names in models.items():
    inputs = [httpclient.InferInput('raw_image', arr.shape, 'UINT8')]
    inputs[0].set_data_from_numpy(arr)
    outputs = [httpclient.InferRequestedOutput(name) for name in output_names]
    try:
        response = client.infer(model_name, inputs=inputs, outputs=outputs)
        num_dets = response.as_numpy('num_dets')[0, 0]
        boxes = response.as_numpy('detection_boxes')
        scores = response.as_numpy('detection_scores')
        classes = response.as_numpy('detection_classes')
        print(f"{model_name}: num_dets={num_dets}")
        print(f"  boxes shape={boxes.shape} scores shape={scores.shape} classes shape={classes.shape}")
        if 'detection_keypoints' in output_names:
            kpts = response.as_numpy('detection_keypoints')
            print(f"  keypoints shape={kpts.shape}")
        if 'detection_masks' in output_names:
            masks = response.as_numpy('detection_masks')
            offsets = response.as_numpy('mask_offsets')
            shapes = response.as_numpy('mask_shapes')
            print(f"  masks shape={masks.shape} offsets shape={offsets.shape} shapes shape={shapes.shape}")
    except Exception as e:
        print(f"{model_name}: FAILED {e}")
print("All dynamic output tests completed.")
