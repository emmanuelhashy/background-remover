<template>
  <div class="min-h-screen bg-gray-100 text-gray-900 p-8">
    <div class="max-w-4xl mx-auto">
      <h1 class="text-4xl font-bold mb-2 text-center text-blue-700">
        In-browser Background Remover Tool
      </h1>
      <h2 class="text-lg font-semibold mb-2 text-center text-gray-600">
        Remove background and download files in real time without relying on a
        remote server, powered by
        <a
          class="underline text-blue-500"
          target="_blank"
          href="https://vuejs.org/"
          >Vue.js</a
        >
        and
        <a
          class="underline text-blue-500"
          target="_blank"
          href="https://github.com/xenova/transformers.js"
          >Transformers.js</a
        >
        with WebGPU support
      </h2>
      <!-- File upload -->
      <div
        class="p-8 mb-8 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors duration-300 ease-in-out"
        :class="{
          'border-green-500 bg-green-100': isDragAccept,
          'border-red-500 bg-red-100': isDragReject,
          'border-blue-500 bg-blue-100': isDragActive,
          'border-gray-400 hover:border-blue-500 hover:bg-gray-200':
            !isDragActive,
        }"
        @dragover.prevent="onDragOver"
        @dragleave="onDragLeave"
        @drop="onDrop"
        @click="triggerFileInput"
      >
        <input
          type="file"
          class="hidden"
          ref="fileInput"
          @change="handleFiles"
          accept="image/*"
          multiple
        />
        <p class="text-lg mb-2">
          {{
            isDragActive
              ? 'Drop the images here...'
              : 'Drag and drop some images here'
          }}
        </p>
        <p class="text-sm text-gray-500">or click to select files</p>
      </div>

      <div class="flex flex-col items-center gap-4 mb-8">
        <button
          @click="processImages"
          :disabled="isProcessing || images.length === 0"
          class="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-100 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-200 text-lg font-semibold"
        >
          {{ isProcessing ? 'Processing...' : 'Process' }}
        </button>

        <div class="flex gap-4">
          <button
            @click="downloadAsZip"
            :disabled="!isDownloadReady"
            class="px-3 py-1 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-gray-100 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-200 text-sm"
          >
            Download as ZIP
          </button>
          <button
            @click="clearAll"
            class="px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-100 transition-colors duration-200 text-sm"
          >
            Clear All
          </button>
        </div>
      </div>

      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        <div v-for="(src, index) in images" :key="index" class="relative group">
          <img
            :src="processedImages[index] || src"
            :alt="'Image ' + (index + 1)"
            class="rounded-lg object-cover w-full h-48"
          />

          <div
            v-if="processedImages[index]"
            class="absolute inset-0 bg-black bg-opacity-70 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg flex items-center justify-center"
          >
            <button
              @click="copyToClipboard(processedImages[index] || src)"
              class="mx-2 px-3 py-1 bg-white text-gray-900 rounded-md hover:bg-gray-200 transition-colors duration-200 text-sm"
            >
              Copy
            </button>
            <button
              @click="downloadImage(processedImages[index] || src)"
              class="mx-2 px-3 py-1 bg-white text-gray-900 rounded-md hover:bg-gray-200 transition-colors duration-200 text-sm"
            >
              Download
            </button>
          </div>

          <button
            @click="removeImage(index)"
            class="absolute top-2 right-2 bg-black bg-opacity-50 text-white w-6 h-6 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 hover:bg-opacity-70"
          >
            &#x2715;
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from '@huggingface/transformers'
import JSZip from 'jszip'
import { saveAs } from 'file-saver'

const images = ref([])
const processedImages = ref([])
const isProcessing = ref(false)
const isDownloadReady = ref(false)
const isDragActive = ref(false)
const isDragAccept = ref(false)
const isDragReject = ref(false)

const modelRef = ref(null)
const processorRef = ref(null)
const fileInput = ref(null)

onMounted(async () => {
  try {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser.')
    }
    const model_id = 'Xenova/modnet'
    env.backends.onnx.wasm.proxy = false
    modelRef.value = await AutoModel.from_pretrained(model_id, {
      device: 'webgpu',
    })
    processorRef.value = await AutoProcessor.from_pretrained(model_id)
  } catch (error) {
    console.error(error)
  }
})

const handleFiles = event => {
  const files = event.target.files
  addImages(files)
}

const addImages = files => {
  for (const file of files) {
    images.value.push(URL.createObjectURL(file))
  }
}

const removeImage = index => {
  images.value.splice(index, 1)
  processedImages.value.splice(index, 1)
}

const processImages = async () => {
  isProcessing.value = true
  processedImages.value = []

  const model = modelRef.value
  const processor = processorRef.value

  for (const image of images.value) {
    const img = await RawImage.fromURL(image)
    const { pixel_values } = await processor(img)
    const { output } = await model({ input: pixel_values })
    const maskData = (
      await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(
        img.width,
        img.height,
      )
    ).data

    const canvas = document.createElement('canvas')
    canvas.width = img.width
    canvas.height = img.height
    const ctx = canvas.getContext('2d')
    ctx.drawImage(img.toCanvas(), 0, 0)

    const pixelData = ctx.getImageData(0, 0, img.width, img.height)
    for (let i = 0; i < maskData.length; ++i) {
      pixelData.data[4 * i + 3] = maskData[i]
    }
    ctx.putImageData(pixelData, 0, 0)
    processedImages.value.push(canvas.toDataURL('image/png'))
  }

  isProcessing.value = false
  isDownloadReady.value = true
}

const downloadAsZip = async () => {
  const zip = new JSZip()
  const promises = images.value.map(
    (image, i) =>
      new Promise(resolve => {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        const img = new Image()
        img.src = processedImages.value[i] || image
        img.onload = () => {
          canvas.width = img.width
          canvas.height = img.height
          ctx.drawImage(img, 0, 0)
          canvas.toBlob(blob => {
            if (blob) {
              zip.file(`image-${i + 1}.png`, blob)
            }
            resolve(null)
          }, 'image/png')
        }
      }),
  )
  await Promise.all(promises)
  const content = await zip.generateAsync({ type: 'blob' })
  saveAs(content, 'images.zip')
}

const clearAll = () => {
  images.value = []
  processedImages.value = []
  isDownloadReady.value = false
}

const onDragOver = event => {
  event.preventDefault()
  isDragActive.value = true
}

const onDragLeave = () => {
  isDragActive.value = false
}

const onDrop = event => {
  event.preventDefault()
  const files = event.dataTransfer.files
  addImages(files)
  isDragActive.value = false
}

const copyToClipboard = async url => {
  const response = await fetch(url)
  const blob = await response.blob()
  const clipboardItem = new ClipboardItem({ [blob.type]: blob })
  await navigator.clipboard.write([clipboardItem])
  console.log('Image copied to clipboard')
}

const downloadImage = url => {
  const a = document.createElement('a')
  a.href = url
  a.download = 'image.png'
  a.click()
}

const triggerFileInput = () => {
  fileInput.value.click()
}
</script>
