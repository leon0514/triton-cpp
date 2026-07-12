import numpy as np
import tritonclient.http as httpclient
import cv2
import os

img_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'bus.jpg')
with open(img_path, 'rb') as f:
    contents = f.read()

nparr = np.frombuffer(contents, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
print('img shape:', img.shape)

batched = np.expand_dims(img, 0)
input_tensor = httpclient.InferInput('raw_image', batched.shape, 'UINT8')
input_tensor.set_data_from_numpy(batched)

output_names = [
    'num_dets', 'detection_boxes', 'detection_scores', 'detection_classes',
    'transform_metadata', 'detection_masks', 'mask_offsets', 'mask_shapes',
]
outputs = [httpclient.InferRequestedOutput(name) for name in output_names]

client = httpclient.InferenceServerClient(url='localhost:48000')
response = client.infer(model_name='yolo11_seg_ensemble', inputs=[input_tensor], outputs=outputs)
print('num_dets:', response.as_numpy('num_dets')[0, 0])
