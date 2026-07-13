<template>
  <div class="app">
    <header class="header">
      <div class="header-left">
        <h1>🚀 Triton Display</h1>
        <span class="subtitle">目标检测 / OBB / Pose / Seg / 分类 可视化</span>
      </div>
      <nav class="nav-tabs">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          class="nav-tab"
          :class="{ active: currentView === tab.key }"
          @click="currentView = tab.key"
        >
          {{ tab.label }}
        </button>
      </nav>
    </header>

    <div class="main">
      <template v-if="currentView === 'inference'">
        <aside class="sidebar">
          <section class="panel">
            <h3>模型</h3>
            <select v-model="selectedModel" :disabled="loading" @change="onModelChange">
              <option value="">请选择 ensemble 模型</option>
              <option v-for="m in models" :key="m.name" :value="m.name">
                {{ m.name }}
              </option>
            </select>
          </section>

          <section class="panel">
            <h3>图片</h3>
            <div
              class="dropzone"
              @click="$refs.fileInput.click()"
              @dragover.prevent
              @drop.prevent="onDrop"
            >
              <span v-if="!imageFile">点击或拖拽上传图片</span>
              <span v-else>{{ imageFile.name }}</span>
            </div>
            <input
              ref="fileInput"
              type="file"
              accept="image/*"
              style="display: none"
              @change="onFileChange"
            />
          </section>

          <template v-if="modelType !== 'classifier'">
            <section class="panel">
              <h3>置信度阈值</h3>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                v-model.number="confThreshold"
              />
              <div class="value">{{ confThreshold.toFixed(2) }}</div>
            </section>

            <section class="panel">
              <h3>显示选项</h3>
              <label><input type="checkbox" v-model="showBoxes" /> 检测框</label>
              <label><input type="checkbox" v-model="showLabels" /> 标签</label>
              <label><input type="checkbox" v-model="showScores" /> 分数</label>
              <label><input type="checkbox" v-model="showPose" /> 姿态骨架</label>
              <label><input type="checkbox" v-model="showSeg" /> 分割 Mask</label>
            </section>

            <section class="panel">
              <h3>标签过滤</h3>
              <div class="label-list">
                <label v-for="label in availableLabels" :key="label.id">
                  <input
                    type="checkbox"
                    :value="label.id"
                    v-model="selectedLabels"
                  />
                  <span
                    class="color-dot"
                    :style="{ backgroundColor: label.color }"
                  ></span>
                  {{ label.name }}
                </label>
              </div>
            </section>
          </template>

          <section class="panel">
            <button
              class="run-btn"
              :disabled="!canRun || loading"
              @click="runInference"
            >
              {{ loading ? '推理中...' : '运行推理' }}
            </button>
            <div v-if="error" class="error">{{ error }}</div>
          </section>
        </aside>

        <div class="content">
          <ClassResult
            v-if="modelType === 'classifier'"
            :result="result"
            :class-names="classNames"
            :image-src="imageUrl"
          />
          <CanvasViewer
            v-else
            :image-src="imageUrl"
            :detections="filteredDetections"
            :model-type="modelType"
            :transform="result?.transform"
            :show-boxes="showBoxes"
            :show-labels="showLabels"
            :show-scores="showScores"
            :show-pose="showPose"
            :show-seg="showSeg"
            :class-names="classNames"
          />
        </div>

        <aside class="rightbar">
          <section class="panel full">
            <h3>结果 JSON</h3>
            <textarea class="json-box" readonly :value="resultJson"></textarea>
            <button class="copy-btn" @click="copyResult">复制结果</button>
          </section>
        </aside>
      </template>

      <Models v-else class="models-view" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import CanvasViewer from './components/CanvasViewer.vue'
import ClassResult from './components/ClassResult.vue'
import Models from './components/Models.vue'
import { fetchModels, fetchLabels, infer } from './api'

const tabs = [
  { key: 'inference', label: '推理' },
  { key: 'models', label: '模型管理' },
]

const currentView = ref('inference')
const models = ref([])
const selectedModel = ref('')
const imageFile = ref(null)
const imageUrl = ref('')
const confThreshold = ref(0.25)
const result = ref(null)
const loading = ref(false)
const error = ref('')

const showBoxes = ref(true)
const showLabels = ref(true)
const showScores = ref(true)
const showPose = ref(true)
const showSeg = ref(true)

const selectedLabels = ref([])
const classNames = ref([])

function getLabelName(classId) {
  return classNames.value[classId] || ''
}

const modelType = computed(() => {
  const name = selectedModel.value.toLowerCase()
  if (name.includes('classifier')) return 'classifier'
  if (name.includes('obb')) return 'obb'
  if (name.includes('pose')) return 'pose'
  if (name.includes('seg')) return 'seg'
  return 'det'
})

const availableLabels = computed(() => {
  const ids = new Set(result.value?.detections?.map((d) => d.class_id) || [])
  return Array.from(ids)
    .sort((a, b) => a - b)
    .filter((id) => {
      const name = getLabelName(id)
      return name && name.trim() !== ''
    })
    .map((id) => ({
      id,
      name: getLabelName(id),
      color: getColor(id),
    }))
})

const filteredDetections = computed(() => {
  if (!result.value?.detections) return []
  return result.value.detections.filter((d) => {
    if (d.score < confThreshold.value) return false
    const name = getLabelName(d.class_id)
    if (!name || name.trim() === '') return false
    if (selectedLabels.value.length > 0 && !selectedLabels.value.includes(d.class_id))
      return false
    return true
  })
})

const resultJson = computed(() => {
  return JSON.stringify(result.value || {}, null, 2)
})

const canRun = computed(() => selectedModel.value && imageFile.value)

function getColor(id) {
  const palette = [
    '#FF5722', '#4CAF50', '#2196F3', '#FFC107', '#9C27B0',
    '#00BCD4', '#E91E63', '#795548', '#607D8B', '#8BC34A',
    '#FF9800', '#3F51B5', '#009688', '#CDDC39', '#F44336',
  ]
  return palette[id % palette.length]
}

function onFileChange(e) {
  const file = e.target.files[0]
  if (file) setImage(file)
}

function onDrop(e) {
  const file = e.dataTransfer.files[0]
  if (file) setImage(file)
}

function setImage(file) {
  imageFile.value = file
  imageUrl.value = URL.createObjectURL(file)
  result.value = null
  selectedLabels.value = []
}

async function onModelChange() {
  classNames.value = []
  selectedLabels.value = []
  if (!selectedModel.value) return
  try {
    const data = await fetchLabels(selectedModel.value)
    classNames.value = data.labels
  } catch (e) {
    console.error('Failed to load labels:', e)
  }
}

async function loadModels() {
  try {
    const data = await fetchModels()
    models.value = data.models.filter((m) => m.ready)
  } catch (e) {
    error.value = e.message
  }
}

async function runInference() {
  if (!canRun.value) return
  loading.value = true
  error.value = ''
  try {
    result.value = await infer(selectedModel.value, imageFile.value, confThreshold.value)
    selectedLabels.value = availableLabels.value.map((l) => l.id)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function copyResult() {
  try {
    await navigator.clipboard.writeText(resultJson.value)
    alert('已复制到剪贴板')
  } catch (e) {
    alert('复制失败')
  }
}

onMounted(loadModels)
</script>

<style>
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f5f5f5;
}

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.header {
  background: #1a237e;
  color: white;
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header h1 {
  margin: 0;
  font-size: 20px;
}

.subtitle {
  opacity: 0.8;
  font-size: 14px;
}

.nav-tabs {
  display: flex;
  gap: 4px;
}

.nav-tab {
  padding: 8px 16px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  background: transparent;
  color: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}

.nav-tab:hover {
  background: rgba(255, 255, 255, 0.15);
}

.nav-tab.active {
  background: white;
  color: #1a237e;
  border-color: white;
}

.main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar,
.rightbar {
  width: 260px;
  background: white;
  border-right: 1px solid #e0e0e0;
  overflow-y: auto;
  padding: 16px;
}

.rightbar {
  border-right: none;
  border-left: 1px solid #e0e0e0;
}

.content {
  flex: 1;
  overflow: auto;
  padding: 16px;
  display: flex;
  justify-content: center;
  align-items: flex-start;
}

.models-view {
  flex: 1;
  overflow: hidden;
  background: #f5f5f5;
}

.panel {
  margin-bottom: 20px;
}

.panel h3 {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #333;
}

.panel select,
.panel input[type='range'] {
  width: 100%;
}

.dropzone {
  border: 2px dashed #ccc;
  border-radius: 6px;
  padding: 20px;
  text-align: center;
  color: #666;
  cursor: pointer;
  transition: border-color 0.2s;
}

.dropzone:hover {
  border-color: #1a237e;
}

.value {
  text-align: center;
  margin-top: 4px;
  font-weight: 500;
}

.panel label {
  display: block;
  margin: 6px 0;
  cursor: pointer;
}

.label-list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid #eee;
  border-radius: 4px;
  padding: 8px;
}

.color-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
}

.run-btn,
.copy-btn {
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 4px;
  background: #1a237e;
  color: white;
  cursor: pointer;
  font-size: 14px;
}

.run-btn:disabled {
  background: #9fa8da;
  cursor: not-allowed;
}

.copy-btn {
  margin-top: 8px;
  background: #4caf50;
}

.error {
  margin-top: 8px;
  color: #d32f2f;
  font-size: 12px;
}

.json-box {
  width: 100%;
  height: calc(100vh - 220px);
  font-family: monospace;
  font-size: 11px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 8px;
  resize: none;
}

.panel.full {
  height: 100%;
}
</style>
