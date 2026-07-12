<template>
  <div class="canvas-wrapper">
    <canvas ref="canvas" @click="onCanvasClick"></canvas>
    <div v-if="!imageSrc" class="placeholder">请先上传图片并运行推理</div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  imageSrc: { type: String, default: '' },
  detections: { type: Array, default: () => [] },
  modelType: { type: String, default: 'det' },
  transform: { type: Array, default: null },
  showBoxes: { type: Boolean, default: true },
  showLabels: { type: Boolean, default: true },
  showScores: { type: Boolean, default: true },
  showPose: { type: Boolean, default: true },
  showSeg: { type: Boolean, default: true },
  classNames: { type: Array, default: () => [] },
})

const canvas = ref(null)
let ctx = null
let currentImage = null

const COCO_SKELETON = [
  [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
  [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
  [2, 4], [3, 5], [4, 6], [5, 7],
]

const palette = [
  '#FF5722', '#4CAF50', '#2196F3', '#FFC107', '#9C27B0',
  '#00BCD4', '#E91E63', '#795548', '#607D8B', '#8BC34A',
  '#FF9800', '#3F51B5', '#009688', '#CDDC39', '#F44336',
]

function getColor(id) {
  return palette[id % palette.length]
}

function applyTransform(x, y, M) {
  if (!M) return [x, y]
  return [
    M[0][0] * x + M[0][1] * y + M[0][2],
    M[1][0] * x + M[1][1] * y + M[1][2],
  ]
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

async function draw() {
  if (!canvas.value) return
  ctx = canvas.value.getContext('2d')
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height)

  if (!props.imageSrc) {
    currentImage = null
    return
  }

  if (!currentImage || currentImage.src !== props.imageSrc) {
    try {
      currentImage = await loadImage(props.imageSrc)
    } catch (e) {
      return
    }
  }

  canvas.value.width = currentImage.naturalWidth
  canvas.value.height = currentImage.naturalHeight
  ctx.drawImage(currentImage, 0, 0)

  props.detections.forEach((det) => drawDetection(det))
}

function drawDetection(det) {
  const name = props.classNames[det.class_id]
  if (!name || name.trim() === '') return

  if (props.modelType === 'obb') {
    drawOBB(det)
  } else {
    drawAABB(det)
  }

  if (props.modelType === 'pose' && det.keypoints && props.showPose) {
    drawPose(det)
  }

  if (props.modelType === 'seg' && props.showSeg && det.mask) {
    drawSeg(det)
  }
}

function drawSeg(det) {
  const [x1, y1, x2, y2] = det.box
  const [x1p, y1p] = applyTransform(x1, y1, props.transform)
  const [x2p, y2p] = applyTransform(x2, y2, props.transform)
  const boxW = x2p - x1p
  const boxH = y2p - y1p
  if (boxW <= 0 || boxH <= 0) return

  const mask = det.mask
  const maskH = mask.length
  const maskW = mask[0]?.length || 0
  if (maskH <= 0 || maskW <= 0) return

  const color = getColor(det.class_id)
  const r = parseInt(color.slice(1, 3), 16)
  const g = parseInt(color.slice(3, 5), 16)
  const b = parseInt(color.slice(5, 7), 16)

  const tmp = document.createElement('canvas')
  tmp.width = maskW
  tmp.height = maskH
  const tctx = tmp.getContext('2d')
  const imgData = tctx.createImageData(maskW, maskH)
  for (let y = 0; y < maskH; y++) {
    for (let x = 0; x < maskW; x++) {
      const idx = (y * maskW + x) * 4
      const alpha = mask[y][x] * 0.5
      imgData.data[idx] = r
      imgData.data[idx + 1] = g
      imgData.data[idx + 2] = b
      imgData.data[idx + 3] = Math.floor(alpha * 255)
    }
  }
  tctx.putImageData(imgData, 0, 0)

  ctx.save()
  ctx.globalCompositeOperation = 'source-over'
  ctx.drawImage(tmp, x1p, y1p, boxW, boxH)
  ctx.restore()
}

function drawAABB(det) {
  if (!props.showBoxes) return
  const [x1, y1, x2, y2] = det.box
  const [x1p, y1p] = applyTransform(x1, y1, props.transform)
  const [x2p, y2p] = applyTransform(x2, y2, props.transform)

  const color = getColor(det.class_id)
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.strokeRect(x1p, y1p, x2p - x1p, y2p - y1p)

  drawLabel(det, x1p, y1p, color)
}

function drawOBB(det) {
  if (!props.showBoxes) return
  const [cx, cy, w, h, angle] = det.box

  const dx = w / 2
  const dy = h / 2
  const corners = [
    [-dx, -dy],
    [dx, -dy],
    [dx, dy],
    [-dx, dy],
  ]

  const cosA = Math.cos(angle)
  const sinA = Math.sin(angle)
  const transformed = corners.map(([x, y]) => {
    const rx = cosA * x - sinA * y + cx
    const ry = sinA * x + cosA * y + cy
    return applyTransform(rx, ry, props.transform)
  })

  const color = getColor(det.class_id)
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(transformed[0][0], transformed[0][1])
  for (let i = 1; i < transformed.length; i++) {
    ctx.lineTo(transformed[i][0], transformed[i][1])
  }
  ctx.closePath()
  ctx.stroke()

  drawLabel(det, transformed[0][0], transformed[0][1], color)
}

function drawLabel(det, x, y, color) {
  if (!props.showLabels && !props.showScores) return

  const name = props.classNames[det.class_id] || `class_${det.class_id}`
  const parts = []
  if (props.showLabels) parts.push(name)
  if (props.showScores) parts.push(det.score.toFixed(2))
  if (parts.length === 0) return

  const text = parts.join(': ')
  ctx.font = '14px sans-serif'
  const metrics = ctx.measureText(text)
  const padding = 4
  const textH = 18

  ctx.fillStyle = color
  ctx.fillRect(x, y - textH - padding, metrics.width + padding * 2, textH + padding)
  ctx.fillStyle = '#fff'
  ctx.fillText(text, x + padding, y - padding)
}

function drawPose(det) {
  const color = getColor(det.class_id)
  const keypoints = det.keypoints.map(([x, y, conf]) => ({
    point: applyTransform(x, y, props.transform),
    conf,
  }))

  ctx.strokeStyle = color
  ctx.lineWidth = 2
  COCO_SKELETON.forEach(([a, b]) => {
    const kpA = keypoints[a - 1]
    const kpB = keypoints[b - 1]
    if (!kpA || !kpB || kpA.conf < 0.3 || kpB.conf < 0.3) return
    ctx.beginPath()
    ctx.moveTo(kpA.point[0], kpA.point[1])
    ctx.lineTo(kpB.point[0], kpB.point[1])
    ctx.stroke()
  })

  keypoints.forEach((kp) => {
    if (kp.conf < 0.3) return
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(kp.point[0], kp.point[1], 3, 0, Math.PI * 2)
    ctx.fill()
  })
}

function onCanvasClick() {
  // future: click to select detection
}

watch(
  () => [
    props.imageSrc,
    props.detections,
    props.modelType,
    props.transform,
    props.showBoxes,
    props.showLabels,
    props.showScores,
    props.showPose,
    props.showSeg,
  ],
  () => {
    nextTick(draw)
  },
  { deep: true }
)
</script>

<style scoped>
.canvas-wrapper {
  position: relative;
  display: inline-block;
  max-width: 100%;
}

canvas {
  max-width: 100%;
  max-height: calc(100vh - 80px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  background: white;
}

.placeholder {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #999;
  font-size: 16px;
}
</style>
