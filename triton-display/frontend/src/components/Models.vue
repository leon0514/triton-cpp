<template>
  <div class="models-page">
    <div class="page-header">
      <div class="page-title">
        <h2>模型管理</h2>
        <span class="page-subtitle">共 {{ sortedModels.length }} 个模型</span>
      </div>
      <div class="toolbar">
        <input
          v-model="search"
          class="search-input"
          type="text"
          placeholder="搜索模型名称..."
        />
        <select v-model="sortKey" class="sort-select">
          <option value="type">按类型排序</option>
          <option value="name">按名称排序</option>
          <option value="state">按状态排序</option>
        </select>
        <button class="sort-order-btn" @click="sortOrder = sortOrder === 'asc' ? 'desc' : 'asc'">
          {{ sortOrder === 'asc' ? '升序 ↑' : '降序 ↓' }}
        </button>
        <button class="refresh-btn" :disabled="loading" @click="loadModels">
          <span class="icon">↻</span>
          {{ loading ? '刷新中...' : '刷新' }}
        </button>
      </div>
    </div>

    <div v-if="error" class="error-alert">{{ error }}</div>

    <div class="table-wrap">
      <table class="model-table">
        <thead>
          <tr>
            <th class="col-status">状态</th>
            <th class="col-name">模型名称</th>
            <th class="col-type">类型</th>
            <th class="col-state">运行状态</th>
            <th class="col-actions">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="m in sortedModels"
            :key="m.name"
            :class="{ ready: m.ready }"
          >
            <td class="col-status">
              <span class="status-dot" :class="m.ready ? 'ready' : 'offline'"></span>
            </td>
            <td class="col-name">
              <strong>{{ m.name }}</strong>
            </td>
            <td class="col-type">
              <span class="type-badge" :class="modelTypeClass(m.name)">
                {{ modelTypeLabel(m.name) }}
              </span>
            </td>
            <td class="col-state">
              <span class="state-badge" :class="m.ready ? 'ready' : 'offline'">
                {{ m.state || 'UNKNOWN' }}
              </span>
            </td>
            <td class="col-actions">
              <button
                class="icon-btn load"
                :disabled="isBusy(m.name) || m.ready"
                :title="'加载模型'"
                @click="onLoad(m.name)"
              >
                ▶
              </button>
              <button
                class="icon-btn unload"
                :disabled="isBusy(m.name) || !m.ready"
                :title="'卸载模型'"
                @click="onUnload(m.name)"
              >
                ⏹
              </button>
              <button
                class="icon-btn config"
                :disabled="isBusy(m.name)"
                :title="'查看配置'"
                @click="onShowConfig(m.name)"
              >
                ⚙
              </button>
            </td>
          </tr>
          <tr v-if="sortedModels.length === 0">
            <td colspan="5" class="empty-cell">未找到匹配的模型</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="selectedModel" class="config-panel" @click.self="closeConfig">
      <div class="config-content">
        <div class="config-header">
          <h3>{{ selectedModel.name }} — 配置信息</h3>
          <button class="close-btn" @click="closeConfig">✕</button>
        </div>
        <pre class="config-json">{{ configJson }}</pre>
        <div v-if="selectedModel.loading" class="config-loading">加载中...</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { fetchAllModels, fetchModelConfig, loadModel, unloadModel } from '../api'

const models = ref([])
const loading = ref(false)
const error = ref('')
const search = ref('')
const sortKey = ref('type')
const sortOrder = ref('asc')
const busyModels = ref(new Set())
const selectedModel = ref(null)

const typeOrder = { CUSTOM: 0, PREPROCESS: 1, MODEL: 2, POSTPROCESS: 3, ENSEMBLE: 4 }
const stateOrder = { READY: 0, LOADING: 1, UNLOADING: 2 }

const filteredModels = computed(() => {
  const s = search.value.trim().toLowerCase()
  if (!s) return models.value
  return models.value.filter((m) => m.name.toLowerCase().includes(s))
})

const sortedModels = computed(() => {
  const list = [...filteredModels.value]
  const order = sortOrder.value === 'asc' ? 1 : -1

  return list.sort((a, b) => {
    if (sortKey.value === 'type') {
      const ta = typeOrder[modelTypeLabel(a.name)] ?? 99
      const tb = typeOrder[modelTypeLabel(b.name)] ?? 99
      if (ta !== tb) return (ta - tb) * order
      // 同类型按状态（READY 在前）
      const sa = stateOrder[a.state] ?? 99
      const sb = stateOrder[b.state] ?? 99
      if (sa !== sb) return sa - sb
      return a.name.localeCompare(b.name)
    }

    if (sortKey.value === 'state') {
      const sa = stateOrder[a.state] ?? 99
      const sb = stateOrder[b.state] ?? 99
      if (sa !== sb) return (sa - sb) * order
      // 同状态按类型
      const ta = typeOrder[modelTypeLabel(a.name)] ?? 99
      const tb = typeOrder[modelTypeLabel(b.name)] ?? 99
      if (ta !== tb) return ta - tb
      return a.name.localeCompare(b.name)
    }

    // sortKey === 'name'
    return a.name.localeCompare(b.name) * order
  })
})

const configJson = computed(() => {
  if (!selectedModel.value) return ''
  return JSON.stringify(selectedModel.value.config || {}, null, 2)
})

function isBusy(name) {
  return busyModels.value.has(name)
}

function modelTypeLabel(name) {
  if (name.endsWith('_ENSEMBLE')) return 'ENSEMBLE'
  if (name.endsWith('_PREPROCESS')) return 'PREPROCESS'
  if (name.endsWith('_POSTPROCESS')) return 'POSTPROCESS'
  if (name === 'CUSTOM_LABELS') return 'CUSTOM'
  return 'MODEL'
}

function modelTypeClass(name) {
  if (name.endsWith('_ENSEMBLE')) return 'ensemble'
  if (name.endsWith('_PREPROCESS')) return 'preprocess'
  if (name.endsWith('_POSTPROCESS')) return 'postprocess'
  if (name === 'CUSTOM_LABELS') return 'labels'
  return 'model'
}

async function loadModels() {
  loading.value = true
  error.value = ''
  try {
    const data = await fetchAllModels()
    models.value = data.models || []
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function onLoad(name) {
  busyModels.value.add(name)
  try {
    await loadModel(name)
    await loadModels()
  } catch (e) {
    error.value = `加载 ${name} 失败: ${e.message}`
  } finally {
    busyModels.value.delete(name)
  }
}

async function onUnload(name) {
  busyModels.value.add(name)
  try {
    await unloadModel(name)
    await loadModels()
  } catch (e) {
    error.value = `卸载 ${name} 失败: ${e.message}`
  } finally {
    busyModels.value.delete(name)
  }
}

async function onShowConfig(name) {
  selectedModel.value = { name, config: null, loading: true }
  try {
    const data = await fetchModelConfig(name)
    selectedModel.value.config = data.config
  } catch (e) {
    selectedModel.value.config = { error: e.message }
  } finally {
    selectedModel.value.loading = false
  }
}

function closeConfig() {
  selectedModel.value = null
}

onMounted(loadModels)
</script>

<style scoped>
.models-page {
  padding: 24px;
  height: 100%;
  overflow-y: auto;
  background: #f8f9fa;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 20px;
}

.page-title h2 {
  margin: 0;
  font-size: 22px;
  color: #1a237e;
}

.page-subtitle {
  color: #666;
  font-size: 13px;
  margin-left: 8px;
}

.toolbar {
  display: flex;
  gap: 12px;
}

.search-input {
  width: 280px;
  padding: 8px 14px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.search-input:focus {
  border-color: #1a237e;
}

.refresh-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: #1a237e;
  color: white;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}

.refresh-btn:hover:not(:disabled) {
  background: #283593;
}

.refresh-btn:disabled {
  background: #9fa8da;
  cursor: not-allowed;
}

.sort-select,
.sort-order-btn {
  padding: 8px 14px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: white;
  color: #333;
  font-size: 14px;
  cursor: pointer;
  outline: none;
}

.sort-select:focus {
  border-color: #1a237e;
}

.sort-order-btn {
  background: #f5f5f5;
  transition: background 0.2s;
}

.sort-order-btn:hover {
  background: #e0e0e0;
}

.icon {
  font-size: 16px;
}

.error-alert {
  margin-bottom: 16px;
  padding: 12px 16px;
  background: #ffebee;
  color: #c62828;
  border-radius: 6px;
  border-left: 4px solid #c62828;
}

.table-wrap {
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
  overflow: hidden;
}

.model-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.model-table thead {
  background: #f1f3f5;
  color: #495057;
}

.model-table th {
  padding: 14px 16px;
  text-align: left;
  font-weight: 600;
  border-bottom: 1px solid #e9ecef;
}

.model-table td {
  padding: 14px 16px;
  border-bottom: 1px solid #f1f3f5;
  vertical-align: middle;
}

.model-table tbody tr:hover {
  background: #f8f9fa;
}

.model-table tbody tr.ready {
  background: #f6fff7;
}

.model-table tbody tr.ready:hover {
  background: #e8f5e9;
}

.col-status {
  width: 60px;
  text-align: center;
}

.col-name {
  min-width: 220px;
}

.col-type {
  width: 120px;
}

.col-state {
  width: 120px;
}

.col-actions {
  width: 150px;
  text-align: right;
}

.status-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.status-dot.ready {
  background: #28a745;
  box-shadow: 0 0 0 3px rgba(40, 167, 69, 0.15);
}

.status-dot.offline {
  background: #adb5bd;
}

.type-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.type-badge.ensemble {
  background: #e3f2fd;
  color: #1565c0;
}

.type-badge.preprocess {
  background: #fff3e0;
  color: #ef6c00;
}

.type-badge.postprocess {
  background: #fce4ec;
  color: #c2185b;
}

.type-badge.labels {
  background: #f3e5f5;
  color: #7b1fa2;
}

.type-badge.model {
  background: #e8f5e9;
  color: #2e7d32;
}

.state-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.state-badge.ready {
  background: #d4edda;
  color: #155724;
}

.state-badge.offline {
  background: #e9ecef;
  color: #495057;
}

.col-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.icon-btn {
  width: 34px;
  height: 34px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: opacity 0.2s, transform 0.1s;
}

.icon-btn:hover:not(:disabled) {
  opacity: 0.85;
  transform: translateY(-1px);
}

.icon-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.icon-btn.load {
  background: #28a745;
  color: white;
}

.icon-btn.unload {
  background: #dc3545;
  color: white;
}

.icon-btn.config {
  background: #17a2b8;
  color: white;
}

.empty-cell {
  text-align: center;
  color: #868e96;
  padding: 40px;
}

.config-panel {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 100;
  padding: 24px;
}

.config-content {
  background: white;
  border-radius: 8px;
  width: 100%;
  max-width: 900px;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
}

.config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #e9ecef;
  background: #f8f9fa;
}

.config-header h3 {
  margin: 0;
  font-size: 16px;
  color: #212529;
}

.close-btn {
  background: transparent;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: #6c757d;
  width: 32px;
  height: 32px;
  border-radius: 4px;
}

.close-btn:hover {
  background: #e9ecef;
}

.config-json {
  flex: 1;
  overflow: auto;
  margin: 0;
  padding: 20px;
  background: #f8f9fa;
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-all;
}

.config-loading {
  padding: 20px;
  text-align: center;
  color: #666;
}
</style>
