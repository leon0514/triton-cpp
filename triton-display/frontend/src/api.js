const API_BASE = import.meta.env.VITE_API_BASE || ''

async function handleError(res) {
  if (!res.ok) {
    let detail = 'Request failed'
    try {
      const err = await res.json()
      detail = err.detail || JSON.stringify(err)
    } catch {
      detail = res.statusText
    }
    throw new Error(detail)
  }
  return res.json()
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`)
  return handleError(res)
}

export async function fetchAllModels() {
  const res = await fetch(`${API_BASE}/api/models/all`)
  return handleError(res)
}

export async function fetchModelConfig(modelName) {
  const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(modelName)}/config`)
  return handleError(res)
}

export async function loadModel(modelName) {
  const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(modelName)}/load`, {
    method: 'POST',
  })
  return handleError(res)
}

export async function unloadModel(modelName) {
  const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(modelName)}/unload`, {
    method: 'POST',
  })
  return handleError(res)
}

export async function fetchLabels(modelName) {
  const res = await fetch(`${API_BASE}/api/labels/${encodeURIComponent(modelName)}`)
  return handleError(res)
}

export async function fetchAllLabels() {
  const res = await fetch(`${API_BASE}/api/labels`)
  return handleError(res)
}

export async function infer(modelName, file, confThreshold = 0.25) {
  const form = new FormData()
  form.append('image', file)
  form.append('conf_threshold', confThreshold)

  const res = await fetch(`${API_BASE}/api/infer/${encodeURIComponent(modelName)}`, {
    method: 'POST',
    body: form,
  })
  return handleError(res)
}
