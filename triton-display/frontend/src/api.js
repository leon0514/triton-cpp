const API_BASE = import.meta.env.VITE_API_BASE || ''

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`)
  if (!res.ok) throw new Error('Failed to fetch models')
  return res.json()
}

export async function fetchLabels(modelName) {
  const res = await fetch(`${API_BASE}/api/labels/${modelName}`)
  if (!res.ok) throw new Error('Failed to fetch labels')
  return res.json()
}

export async function fetchAllLabels() {
  const res = await fetch(`${API_BASE}/api/labels`)
  if (!res.ok) throw new Error('Failed to fetch labels')
  return res.json()
}

export async function infer(modelName, file, confThreshold = 0.25) {
  const form = new FormData()
  form.append('image', file)
  form.append('conf_threshold', confThreshold)

  const res = await fetch(`${API_BASE}/api/infer/${modelName}`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Inference failed')
  }
  return res.json()
}
