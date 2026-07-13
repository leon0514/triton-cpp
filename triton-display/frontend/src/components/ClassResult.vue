<template>
  <div class="class-result">
    <div class="image-preview">
      <img v-if="imageSrc" :src="imageSrc" alt="uploaded" />
      <div v-else class="placeholder">请先上传图片</div>
    </div>

    <div class="predictions">
      <h3>分类结果</h3>
      <div v-if="items.length === 0" class="empty">暂无结果</div>
      <div
        v-for="(item, idx) in items"
        :key="idx"
        class="pred-row"
      >
        <span class="class-name">{{ item.name }}</span>
        <div class="bar-bg">
          <div
            class="bar"
            :style="{ width: item.percent + '%', backgroundColor: item.color }"
          ></div>
        </div>
        <span class="score">{{ item.percent }}%</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  result: { type: Object, default: null },
  classNames: { type: Array, default: () => [] },
  imageSrc: { type: String, default: '' },
})

function getColor(id) {
  const palette = [
    '#FF5722', '#4CAF50', '#2196F3', '#FFC107', '#9C27B0',
    '#00BCD4', '#E91E63', '#795548', '#607D8B', '#8BC34A',
    '#FF9800', '#3F51B5', '#009688', '#CDDC39', '#F44336',
  ]
  return palette[id % palette.length]
}

const items = computed(() => {
  if (!props.result) return []
  const classes = props.result.classes || []
  const scores = props.result.scores || []
  return classes.map((classId, i) => {
    const score = scores[i] || 0
    const name = props.classNames[classId]?.trim() || `class ${classId}`
    return {
      name,
      percent: parseFloat((score * 100).toFixed(1)),
      color: getColor(classId),
    }
  })
})
</script>

<style scoped>
.class-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  max-width: 720px;
  width: 100%;
}

.image-preview {
  display: flex;
  justify-content: center;
  align-items: center;
  max-width: 100%;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 12px;
}

.image-preview img {
  max-width: 100%;
  max-height: 55vh;
  border-radius: 6px;
  object-fit: contain;
}

.placeholder {
  width: 320px;
  height: 240px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #9aa0a6;
  background: #f8f9fa;
  border-radius: 6px;
  font-size: 14px;
}

.predictions {
  width: 100%;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 20px;
}

.predictions h3 {
  margin: 0 0 16px 0;
  font-size: 14px;
  font-weight: 600;
  color: #5f6368;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.empty {
  color: #9aa0a6;
  text-align: center;
  padding: 16px 0;
  font-size: 14px;
}

.pred-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 14px 0;
  font-size: 14px;
}

.class-name {
  width: 120px;
  flex-shrink: 0;
  text-align: right;
  color: #3c4043;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-weight: 500;
}

.bar-bg {
  flex: 1;
  height: 16px;
  background: #e8eaed;
  border-radius: 8px;
  overflow: hidden;
}

.bar {
  height: 100%;
  border-radius: 8px;
  transition: width 0.4s ease;
}

.score {
  width: 52px;
  flex-shrink: 0;
  text-align: right;
  font-weight: 600;
  color: #5f6368;
  font-size: 13px;
}
</style>
