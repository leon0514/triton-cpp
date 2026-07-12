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
  gap: 24px;
  max-width: 800px;
  width: 100%;
}

.image-preview {
  display: flex;
  justify-content: center;
  align-items: center;
  max-width: 100%;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 12px;
}

.image-preview img {
  max-width: 100%;
  max-height: 60vh;
  border-radius: 4px;
  object-fit: contain;
}

.placeholder {
  width: 320px;
  height: 240px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #888;
  background: #f5f5f5;
  border-radius: 4px;
}

.predictions {
  width: 100%;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 20px;
}

.predictions h3 {
  margin: 0 0 16px 0;
  font-size: 16px;
  color: #333;
}

.empty {
  color: #888;
  text-align: center;
  padding: 12px 0;
}

.pred-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 12px 0;
  font-size: 14px;
}

.class-name {
  width: 120px;
  flex-shrink: 0;
  text-align: right;
  color: #333;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.bar-bg {
  flex: 1;
  height: 18px;
  background: #e0e0e0;
  border-radius: 9px;
  overflow: hidden;
}

.bar {
  height: 100%;
  border-radius: 9px;
  transition: width 0.3s ease;
}

.score {
  width: 60px;
  flex-shrink: 0;
  text-align: right;
  font-weight: 500;
  color: #555;
}
</style>
