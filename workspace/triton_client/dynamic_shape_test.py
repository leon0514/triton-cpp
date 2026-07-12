import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import os

img_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'bus.jpg')
img = Image.open(img_path).convert('RGB')
arr = np.array(img, dtype=np.uint8)
arr = np.expand_dims(arr, axis=0)

inputs = [httpclient.InferInput('raw_image', arr.shape, 'UINT8')]
inputs[0].set_data_from_numpy(arr)

outputs = [
    httpclient.InferRequestedOutput('num_dets'),
    httpclient.InferRequestedOutput('detection_boxes'),
    httpclient.InferRequestedOutput('detection_scores'),
    httpclient.InferRequestedOutput('detection_classes'),
    httpclient.InferRequestedOutput('detection_masks'),
    httpclient.InferRequestedOutput('mask_offsets'),
    httpclient.InferRequestedOutput('mask_shapes'),
]

client = httpclient.InferenceServerClient(url='localhost:48000')
response = client.infer('yolo11_seg_ensemble', inputs=inputs, outputs=outputs)

print('num_dets:', response.as_numpy('num_dets').shape)
print('detection_boxes:', response.as_numpy('detection_boxes').shape)
print('detection_scores:', response.as_numpy('detection_scores').shape)
print('detection_classes:', response.as_numpy('detection_classes').shape)
print('detection_masks:', response.as_numpy('detection_masks').shape)
print('mask_offsets:', response.as_numpy('mask_offsets').shape)
print('mask_shapes:', response.as_numpy('mask_shapes').shape)
