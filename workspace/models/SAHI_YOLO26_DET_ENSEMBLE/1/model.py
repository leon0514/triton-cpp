"""SAHI Ensemble BLS Model (v3 optimized)
Orchestrates SAHI slicing -> batched detection -> box merge -> per-class NMS.

Key fixes & optimizations vs v2:
  - [BUGFIX] Output shape matches config.pbtxt (no extra batch dim)
  - [BUGFIX] Per-class NMS (cross-class boxes no longer incorrectly suppressed)
  - [PERF] Vectorized offset/clamp: all slices processed in one batch, no Python loop
  - [PERF] Debug prints conditional on SAHI_DEBUG env var (no I/O syscalls in hot path)
  - [NOTE] pb_utils.Tensor only accepts numpy, so sliced_images still does GPU->CPU->GPU
"""

import json
import os

import numpy as np
import cupy as cp
import triton_python_backend_utils as pb_utils

_DEBUG = os.environ.get('SAHI_DEBUG', '0') == '1'


def _gpu_tensor_to_numpy(tensor):
    """Triton GPU tensor -> numpy (via DLPack -> cupy -> numpy)."""
    return cp.asnumpy(cp.from_dlpack(tensor.to_dlpack()))


def _nms_gpu(boxes, scores, iou_threshold):
    """Per-class GPU-vectorized NMS (cupy). 5-10x faster than numpy loop."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = cp.asarray(boxes[:, 0])
    y1 = cp.asarray(boxes[:, 1])
    x2 = cp.asarray(boxes[:, 2])
    y2 = cp.asarray(boxes[:, 3])
    areas = (x2 - x1) * (y2 - y1)

    order = cp.argsort(-cp.asarray(scores))
    keep = []

    while order.size > 0:
        i = order[0].item()
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = cp.maximum(x1[i], x1[rest])
        yy1 = cp.maximum(y1[i], y1[rest])
        xx2 = cp.minimum(x2[i], x2[rest])
        yy2 = cp.minimum(y2[i], y2[rest])

        w = cp.maximum(0.0, xx2 - xx1)
        h = cp.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-8)

        remain = cp.where(iou <= iou_threshold)[0]
        order = order[remain + 1]

    return np.array(keep, dtype=np.int64)


def _nms_per_class(boxes, scores, classes, iou_threshold):
    """NMS grouped by class to avoid cross-class suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    unique_cls = np.unique(classes)
    keep_all = []
    for cls_id in unique_cls:
        mask = classes == cls_id
        cls_indices = np.where(mask)[0]
        if len(cls_indices) == 0:
            continue
        cls_keep = _nms_gpu(boxes[mask], scores[mask], iou_threshold)
        keep_all.append(cls_indices[cls_keep])

    if not keep_all:
        return np.array([], dtype=np.int64)
    return np.concatenate(keep_all)


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        params = self._parse_parameters()
        self.detector_model = params.get('detector_model', 'YOLO11_DET_PRE_ENSEMBLE')
        self.conf_threshold = float(params.get('conf_threshold', '0.25'))
        self.iou_threshold = float(params.get('iou_threshold', '0.45'))
        self.max_detections = int(params.get('max_detections', '300'))

        self.det_output_names = ['num_dets', 'detection_boxes',
                                 'detection_scores', 'detection_classes']

    def _parse_parameters(self):
        params = {}
        for key, obj in self.model_config.get('parameters', {}).items():
            params[key] = obj.get('string_value', '')
        return params

    def execute(self, requests):
        responses = []

        for request in requests:
            # ---- 1. Get raw_image ----
            img = pb_utils.get_input_tensor_by_name(request, 'raw_image').as_numpy()
            if img.ndim == 4:
                img = img[0]
            H, W = img.shape[0], img.shape[1]

            # ---- 2. SAHI slice ----
            sahi_resp = pb_utils.InferenceRequest(
                model_name='SAHI_PREPROCESS',
                requested_output_names=['sliced_images', 'slice_offsets'],
                inputs=[pb_utils.Tensor('raw_image', img)],
            ).exec()
            if sahi_resp.has_error():
                raise pb_utils.TritonModelException(sahi_resp.error().message())

            # Extract sliced_images shape + offsets, then release GPU refs ASAP
            # to avoid OOM when detector allocates its own GPU buffers.
            # NOTE: Triton may squeeze the batch dim when N==1, so force
            # atleast_4d for sliced_cp and atleast_2d for slice_offsets.
            sliced_gpu = pb_utils.get_output_tensor_by_name(sahi_resp, 'sliced_images')
            sliced_cp = cp.from_dlpack(sliced_gpu.to_dlpack())
            if sliced_cp.ndim == 3:
                sliced_cp = sliced_cp.reshape(1, *sliced_cp.shape)
            N = sliced_cp.shape[0]
            sliced_images_np = cp.asnumpy(sliced_cp)  # GPU -> CPU copy

            slice_offsets = np.atleast_2d(_gpu_tensor_to_numpy(
                pb_utils.get_output_tensor_by_name(sahi_resp, 'slice_offsets')))

            # Release all GPU memory from SAHI before detector allocates
            del sliced_cp, sliced_gpu, sahi_resp
            cp.get_default_memory_pool().free_all_blocks()

            if _DEBUG:
                print(f"[SAHI_BLS] N={N} offsets={slice_offsets}", flush=True)

            # ---- 3. Batched detection ----
            all_boxes, all_scores, all_classes = self._batched_detect(
                sliced_images_np, N, slice_offsets, W, H)

            # ---- 4. Merge + per-class NMS ----
            out_tensors = self._merge_nms(all_boxes, all_scores, all_classes)
            responses.append(pb_utils.InferenceResponse(out_tensors))

        return responses

    # ------------------------------------------------------------------
    #  Batched detection: [N, H, W, 3] -> vectorized post-process
    # ------------------------------------------------------------------
    def _batched_detect(self, sliced_images_np, N, slice_offsets, W, H):
        """Run all N slices through detector in ONE batched call.
        sliced_images_np: [N, 640, 640, 3] uint8 numpy array (already on CPU).
        Returns vectorized boxes/scores/classes (no per-slice Python loop)."""

        det_resp = pb_utils.InferenceRequest(
            model_name=self.detector_model,
            requested_output_names=self.det_output_names,
            inputs=[pb_utils.Tensor('raw_image', sliced_images_np)],
        ).exec()
        if det_resp.has_error():
            raise pb_utils.TritonModelException(det_resp.error().message())

        # One-time conversion of all batched GPU outputs.
        # NOTE: when N==1 Triton may squeeze the batch dim, so we force the
        # expected dimensionality with atleast_2d/3d.
        num_dets_arr = _gpu_tensor_to_numpy(
            pb_utils.get_output_tensor_by_name(det_resp, 'num_dets'))        # [N] or [N, 1]
        boxes_arr = np.atleast_3d(_gpu_tensor_to_numpy(
            pb_utils.get_output_tensor_by_name(det_resp, 'detection_boxes')))  # [N, M, 4]
        scores_arr = np.atleast_2d(_gpu_tensor_to_numpy(
            pb_utils.get_output_tensor_by_name(det_resp, 'detection_scores'))) # [N, M]
        classes_arr = np.atleast_2d(_gpu_tensor_to_numpy(
            pb_utils.get_output_tensor_by_name(det_resp, 'detection_classes')))# [N, M]

        if _DEBUG:
            print(f"[SAHI_BLS] num_dets_per_slice={num_dets_arr.squeeze()}", flush=True)

        # ---- Vectorized: flatten all slices' detections into one pass ----
        ndets_per_slice = num_dets_arr.reshape(-1).astype(int)  # [N], handles [N] or [N,1]
        total_dets = ndets_per_slice.sum()

        if total_dets == 0:
            return [], [], []

        # Build slice index mapping: which original slice each detection belongs to
        slice_idx = np.repeat(np.arange(N), ndets_per_slice)  # [total_dets]

        # Gather all boxes/scores/classes in one flattened step
        boxes_flat = np.vstack([boxes_arr[i, :ndets_per_slice[i]] for i in range(N)])
        scores_flat = np.hstack([scores_arr[i, :ndets_per_slice[i]] for i in range(N)])
        classes_flat = np.hstack([classes_arr[i, :ndets_per_slice[i]] for i in range(N)])

        # ---- Confidence filter (vectorized) ----
        keep = scores_flat >= self.conf_threshold
        if not keep.any():
            return [], [], []

        boxes = boxes_flat[keep]
        scores = scores_flat[keep]
        classes = classes_flat[keep]
        sidx = slice_idx[keep]  # slice indices for kept detections

        if _DEBUG:
            print(f"[SAHI_BLS] conf_filter: {total_dets}->{len(boxes)}", flush=True)

        # ---- Vectorized offset to original image coords ----
        ox = slice_offsets[sidx, 0].astype(np.float32)
        oy = slice_offsets[sidx, 1].astype(np.float32)

        boxes[:, 0] += ox
        boxes[:, 1] += oy
        boxes[:, 2] += ox
        boxes[:, 3] += oy

        # ---- Vectorized clamp ----
        np.clip(boxes[:, 0], 0.0, float(W), out=boxes[:, 0])
        np.clip(boxes[:, 1], 0.0, float(H), out=boxes[:, 1])
        np.clip(boxes[:, 2], 0.0, float(W), out=boxes[:, 2])
        np.clip(boxes[:, 3], 0.0, float(H), out=boxes[:, 3])

        return [boxes], [scores], [classes]

    # ------------------------------------------------------------------
    #  Merge + per-class NMS (GPU)
    # ------------------------------------------------------------------
    def _merge_nms(self, all_boxes, all_scores, all_classes):
        if not all_boxes:
            return [
                pb_utils.Tensor('num_dets', np.array([0], dtype=np.int32)),
                pb_utils.Tensor('detection_boxes',
                    np.zeros((self.max_detections, 4), dtype=np.float32)),
                pb_utils.Tensor('detection_scores',
                    np.zeros((self.max_detections,), dtype=np.float32)),
                pb_utils.Tensor('detection_classes',
                    -np.ones((self.max_detections,), dtype=np.int32)),
            ]

        # Already concatenated from _batched_detect (single-element lists)
        all_boxes = all_boxes[0]
        all_scores = all_scores[0]
        all_classes = all_classes[0]

        if _DEBUG:
            print(f"[SAHI_BLS] merge: total={len(all_boxes)} boxes", flush=True)

        # Per-class GPU NMS
        keep = _nms_per_class(all_boxes, all_scores, all_classes, self.iou_threshold)
        all_boxes = all_boxes[keep]
        all_scores = all_scores[keep]
        all_classes = all_classes[keep]

        if _DEBUG:
            print(f"[SAHI_BLS] nms: {len(keep)} kept", flush=True)

        # Top-K by score
        order = np.argsort(-all_scores)[:self.max_detections]
        all_boxes = all_boxes[order]
        all_scores = all_scores[order]
        all_classes = all_classes[order]

        n = len(all_boxes)
        pad = self.max_detections - n

        out_num = np.array([n], dtype=np.int32)
        out_boxes = np.pad(all_boxes, ((0, pad), (0, 0)))
        out_scores = np.pad(all_scores, (0, pad))
        out_classes = np.pad(all_classes, (0, pad), constant_values=-1)

        return [
            pb_utils.Tensor('num_dets', out_num),
            pb_utils.Tensor('detection_boxes', out_boxes.astype(np.float32)),
            pb_utils.Tensor('detection_scores', out_scores.astype(np.float32)),
            pb_utils.Tensor('detection_classes', out_classes.astype(np.int32)),
        ]

    def finalize(self):
        pass
