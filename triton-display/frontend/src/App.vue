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
      <div v-show="currentView === 'inference'" class="inference-layout">
        <aside class="sidebar">
          <section class="panel">
            <h3>模型</h3>
            <div class="custom-select" v-click-outside="closeModelSelect">
              <div
                class="custom-select-trigger"
                :class="{ open: modelSelectOpen, placeholder: !selectedModel }"
                @click="modelSelectOpen = !modelSelectOpen"
              >
                <span :title="selectedModel || '请选择 ensemble 模型'">
                  {{ selectedModel || '请选择 ensemble 模型' }}
                </span>
                <span class="arrow">▼</span>
              </div>
              <transition name="fade-slide">
                <ul v-show="modelSelectOpen" class="custom-select-options">
                  <li
                    v-for="m in models"
                    :key="m.name"
                    :class="{ active: m.name === selectedModel }"
                    @click="selectModel(m.name)"
                  >
                    {{ m.name }}
                  </li>
                </ul>
              </transition>
            </div>
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
              <input
                v-model="labelSearch"
                type="text"
                placeholder="搜索标签..."
                class="label-search"
              />
              <div class="label-actions">
                <button @click="selectAllLabels">全选</button>
                <button @click="deselectAllLabels">清空</button>
              </div>
              <div class="label-list">
                <label v-for="label in filteredLabels" :key="label.id">
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
      </div>

      <Models v-show="currentView !== 'inference'" class="models-view" />
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
const loading = ref(false)
const error = ref('')

const showBoxes = ref(true)
const showLabels = ref(true)
const showScores = ref(true)
const showPose = ref(true)
const showSeg = ref(true)

const labelSearch = ref('')

const resultsCache = ref({})
const classNamesCache = ref({})
const selectedLabelsCache = ref({})

const result = computed({
  get: () => (selectedModel.value ? resultsCache.value[selectedModel.value] ?? null : null),
  set: (v) => {
    if (selectedModel.value) resultsCache.value[selectedModel.value] = v
  },
})

const classNames = computed({
  get: () => (selectedModel.value ? classNamesCache.value[selectedModel.value] ?? [] : []),
  set: (v) => {
    if (selectedModel.value) classNamesCache.value[selectedModel.value] = v
  },
})

const selectedLabels = computed({
  get: () => (selectedModel.value ? selectedLabelsCache.value[selectedModel.value] ?? [] : []),
  set: (v) => {
    if (selectedModel.value) selectedLabelsCache.value[selectedModel.value] = v
  },
})

const modelSelectOpen = ref(false)

const vClickOutside = {
  mounted(el, binding) {
    el._clickOutside = (e) => {
      if (!el.contains(e.target)) binding.value()
    }
    document.addEventListener('click', el._clickOutside)
  },
  unmounted(el) {
    document.removeEventListener('click', el._clickOutside)
  },
}

function selectModel(name) {
  if (name === selectedModel.value) {
    modelSelectOpen.value = false
    return
  }
  selectedModel.value = name
  modelSelectOpen.value = false
  onModelChange()
}

function closeModelSelect() {
  modelSelectOpen.value = false
}

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

const filteredLabels = computed(() => {
  const kw = labelSearch.value.trim().toLowerCase()
  if (!kw) return availableLabels.value
  return availableLabels.value.filter((l) =>
    l.name.toLowerCase().includes(kw)
  )
})

function selectAllLabels() {
  const current = new Set(selectedLabels.value)
  filteredLabels.value.forEach((l) => current.add(l.id))
  selectedLabels.value = Array.from(current)
}

function deselectAllLabels() {
  const toRemove = new Set(filteredLabels.value.map((l) => l.id))
  selectedLabels.value = selectedLabels.value.filter((id) => !toRemove.has(id))
}

const filteredDetections = computed(() => {
  if (!result.value?.detections) return []
  return result.value.detections.filter((d) => {
    if (d.score < confThreshold.value) return false
    const name = getLabelName(d.class_id)
    if (!name || name.trim() === '') return false
    if (!selectedLabels.value.includes(d.class_id)) return false
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
  if (!selectedModel.value) return
  if (classNames.value.length === 0) {
    try {
      const data = await fetchLabels(selectedModel.value)
      classNames.value = data.labels
    } catch (e) {
      console.error('Failed to load labels:', e)
    }
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
  const text = resultJson.value
  if (!text) {
    alert('没有可复制的结果')
    return
  }
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text)
      alert('已复制到剪贴板')
    } else {
      fallbackCopy(text)
    }
  } catch (e) {
    fallbackCopy(text)
  }
}

function fallbackCopy(text) {
  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.style.position = 'fixed'
  textarea.style.left = '-9999px'
  textarea.style.top = '0'
  document.body.appendChild(textarea)
  textarea.focus()
  textarea.select()
  let ok = false
  try {
    ok = document.execCommand('copy')
  } catch (e) {
    ok = false
  }
  document.body.removeChild(textarea)
  alert(ok ? '已复制到剪贴板' : '复制失败')
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
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.header h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.subtitle {
  opacity: 0.75;
  font-size: 13px;
  font-weight: 400;
}

.nav-tabs {
  display: flex;
  gap: 6px;
  background: rgba(255, 255, 255, 0.12);
  padding: 4px;
  border-radius: 8px;
}

.nav-tab {
  padding: 7px 16px;
  border: none;
  background: transparent;
  color: rgba(255, 255, 255, 0.85);
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.2s;
}

.nav-tab:hover {
  color: white;
  background: rgba(255, 255, 255, 0.1);
}

.nav-tab.active {
  background: white;
  color: #1a237e;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
}

.main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.inference-layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.models-view {
  flex: 1;
  overflow: hidden;
  background: #f5f5f5;
}

.sidebar,
.rightbar {
  width: 240px;
  background: #f8f9fa;
  border-right: 1px solid #e8eaed;
  overflow-y: auto;
  padding: 16px;
}

.rightbar {
  border-right: none;
  border-left: 1px solid #e8eaed;
}

.content {
  flex: 1;
  overflow: auto;
  padding: 20px;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  background: #eef0f4;
}

.panel {
  background: white;
  border-radius: 10px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
  padding: 14px;
  margin-bottom: 14px;
}

.panel h3 {
  margin: 0 0 12px 0;
  font-size: 12px;
  font-weight: 600;
  color: #5f6368;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.panel input[type='range'] {
  width: 100%;
  accent-color: #1a237e;
}

.custom-select {
  position: relative;
  font-size: 13px;
}

.custom-select-trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 9px 11px;
  border: 1px solid #dadce0;
  border-radius: 7px;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
  color: #202124;
}

.custom-select-trigger:hover {
  border-color: #1a237e;
}

.custom-select-trigger.open {
  border-color: #1a237e;
  box-shadow: 0 0 0 3px rgba(26, 35, 126, 0.1);
}

.custom-select-trigger.placeholder {
  color: #80868b;
}

.custom-select-trigger span:first-child {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 8px;
}

.custom-select-trigger .arrow {
  font-size: 10px;
  color: #5f6368;
  transition: transform 0.2s;
  flex-shrink: 0;
}

.custom-select-trigger.open .arrow {
  transform: rotate(180deg);
}

.custom-select-options {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  right: 0;
  max-height: 260px;
  overflow-y: auto;
  background: white;
  border: 1px solid #e8eaed;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  list-style: none;
  margin: 0;
  padding: 6px;
  z-index: 100;
}

.custom-select-options li {
  padding: 8px 10px;
  border-radius: 5px;
  cursor: pointer;
  color: #3c4043;
  line-height: 1.4;
  word-break: break-all;
  transition: background 0.15s;
}

.custom-select-options li:hover,
.custom-select-options li.active {
  background: #f3f4f9;
  color: #1a237e;
}

.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.fade-slide-enter-from,
.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.dropzone {
  border: 2px dashed #dadce0;
  border-radius: 8px;
  padding: 18px 12px;
  text-align: center;
  color: #5f6368;
  cursor: pointer;
  transition: all 0.2s;
  background: #fafafa;
  font-size: 13px;
}

.dropzone:hover {
  border-color: #1a237e;
  background: #f3f4f9;
}

.value {
  text-align: center;
  margin-top: 6px;
  font-weight: 600;
  color: #1a237e;
  font-size: 13px;
}

.panel label {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0;
  cursor: pointer;
  font-size: 13px;
  color: #3c4043;
}

.panel label input[type='checkbox'] {
  accent-color: #1a237e;
  width: 14px;
  height: 14px;
  margin: 0;
}

.label-list {
  max-height: 180px;
  overflow-y: auto;
  border: 1px solid #e8eaed;
  border-radius: 6px;
  padding: 6px;
  background: #fafafa;
}

.label-list label {
  padding: 4px 6px;
  border-radius: 4px;
  margin: 2px 0;
}

.label-list label:hover {
  background: #f0f1f4;
}

.label-search {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid #dadce0;
  border-radius: 6px;
  font-size: 13px;
  margin-bottom: 8px;
}

.label-actions {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}

.label-actions button {
  flex: 1;
  padding: 6px 0;
  border: 1px solid #dadce0;
  border-radius: 6px;
  background: white;
  cursor: pointer;
  font-size: 12px;
  color: #5f6368;
  transition: all 0.15s;
}

.label-actions button:hover {
  border-color: #1a237e;
  color: #1a237e;
  background: #f3f4f9;
}

.color-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 2px;
  flex-shrink: 0;
}

.run-btn,
.copy-btn {
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 6px;
  background: #1a237e;
  color: white;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  box-shadow: 0 1px 2px rgba(26, 35, 126, 0.2);
  transition: background 0.2s, transform 0.1s;
}

.run-btn:hover:not(:disabled) {
  background: #283593;
  transform: translateY(-1px);
}

.run-btn:disabled {
  background: #9fa8da;
  cursor: not-allowed;
  box-shadow: none;
}

.copy-btn {
  margin-top: 10px;
  background: #4caf50;
  box-shadow: 0 1px 2px rgba(76, 175, 80, 0.2);
}

.copy-btn:hover {
  background: #43a047;
}

.error {
  margin-top: 10px;
  padding: 8px 10px;
  border-radius: 6px;
  background: #ffebee;
  color: #c62828;
  font-size: 12px;
}

.json-box {
  width: 100%;
  height: calc(100vh - 240px);
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 11px;
  line-height: 1.5;
  border: 1px solid #e8eaed;
  border-radius: 6px;
  padding: 10px;
  resize: none;
  background: #fafafa;
  color: #202124;
}

.panel.full {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.panel.full .json-box {
  flex: 1;
  height: auto;
}
</style>
