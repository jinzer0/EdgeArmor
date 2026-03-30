import { clampBbox } from "./image_utils.js";
import { FaceDetector, FilesetResolver } from "./vendor/mediapipe/vision_bundle.mjs";

const MEDIAPIPE_MODEL_FILE = "blaze_face_short_range.tflite";
const MEDIAPIPE_WASM_DIR = chrome.runtime.getURL("vendor/mediapipe/wasm");
const MEDIAPIPE_MODEL_URL = chrome.runtime.getURL(`models/${MEDIAPIPE_MODEL_FILE}`);

let detectorPromise = null;
let modelBufferPromise = null;

export async function detectFacesWithMediaPipe(image, options = {}) {
  const detector = await getFaceDetector(options);
  const result = detector.detect(image);
  return normalizeMediaPipeDetections(result?.detections || [], image);
}

async function getFaceDetector(options) {
  if (!detectorPromise) {
    detectorPromise = createFaceDetector(options).catch((error) => {
      detectorPromise = null;
      throw error;
    });
  }

  return detectorPromise;
}

async function createFaceDetector(options) {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_DIR);
  const modelAssetBuffer = await loadModelBuffer();
  return FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetBuffer,
      delegate: "CPU",
    },
    runningMode: "IMAGE",
    minDetectionConfidence: options.minDetectionConfidence,
    minSuppressionThreshold: options.minSuppressionThreshold,
  });
}

async function loadModelBuffer() {
  if (!modelBufferPromise) {
    modelBufferPromise = fetch(MEDIAPIPE_MODEL_URL)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`MediaPipe face detector model load failed: HTTP ${response.status}`);
        }

        const buffer = await response.arrayBuffer();
        return new Uint8Array(buffer);
      })
      .catch((error) => {
        modelBufferPromise = null;
        throw error;
      });
  }

  return modelBufferPromise;
}

function normalizeMediaPipeDetections(detections, image) {
  return detections
    .map((detection) => normalizeMediaPipeDetection(detection, image))
    .filter(Boolean)
    .sort((left, right) => right.confidence - left.confidence);
}

function normalizeMediaPipeDetection(detection, image) {
  const boundingBox = detection?.boundingBox;
  const category = detection?.categories?.[0];
  if (!boundingBox || !category) {
    return null;
  }

  const x1 = boundingBox.originX;
  const y1 = boundingBox.originY;
  const x2 = boundingBox.originX + boundingBox.width;
  const y2 = boundingBox.originY + boundingBox.height;
  const bbox = clampBbox([x1, y1, x2, y2], image.naturalWidth, image.naturalHeight);
  const landmarks = (detection.keypoints || []).map((keypoint) => [
    keypoint.x * image.naturalWidth,
    keypoint.y * image.naturalHeight,
  ]);

  return {
    bbox,
    confidence: category.score || 0,
    landmarks,
  };
}
