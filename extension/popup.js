import {
  cropFaceToCanvas,
  fitCanvas,
  letterboxImageToCanvas,
  resizeCanvas,
  tensorFromCanvas,
} from "./image_utils.js";
import { detectFacesWithOnnx } from "./onnx_face_detector.js";

const CONFIG = {
  runtimeMode: "browser",
  localServerUrl: "http://127.0.0.1:8765/analyze",
  classifierInputSize: 256,
  minConfidence: 0.5,
  minFaceSize: 64,
  topK: 3,
  margin: 0.4,
  fakeThreshold: 0.5,
  mean: [0.48145466, 0.4578275, 0.40821073],
  std: [0.26862954, 0.26130258, 0.27577711],
  hfMean: [0.5, 0.5, 0.5],
  hfStd: [0.5, 0.5, 0.5],
};

const HF_DEEPFAKE_CLASS_INDEX = 1;
const CLASSIFIER_ONNX_FILE = "model.onnx";

const MODEL_URLS = {
  classifier: chrome.runtime.getURL(`models/${CLASSIFIER_ONNX_FILE}`),
};

const statusText = document.getElementById("status-text");
const timingChip = document.getElementById("timing-chip");
const analyzeButton = document.getElementById("analyze-button");
const resetButton = document.getElementById("reset-button");
const imageInput = document.getElementById("image-input");
const previewCanvas = document.getElementById("preview-canvas");
const previewContext = previewCanvas.getContext("2d");
const summaryPanel = document.getElementById("summary-panel");
const faceList = document.getElementById("face-list");
let ort = null;

const state = {
  file: null,
  image: null,
  classifierSessionPromise: null,
  classifierSession: null,
};

imageInput.addEventListener("change", handleFileChange);
analyzeButton.addEventListener("click", () => {
  void analyzeSelectedImage();
});
resetButton.addEventListener("click", resetUi);

resetUi();

async function handleFileChange(event) {
  const [file] = event.target.files || [];
  if (!file) {
    return;
  }

  state.file = file;
  state.image = await loadImageFromFile(file);
  analyzeButton.disabled = false;
  drawPreviewImage(state.image);
  setStatus(
    `이미지 로드 완료: ${file.name}. 분석 버튼을 눌러 주세요.`,
    "ready",
  );
}

async function analyzeSelectedImage() {
  if (!state.image || !state.file) {
    setStatus("먼저 이미지를 추가해 주세요.", "idle");
    return;
  }

  if (CONFIG.runtimeMode === "local_server") {
    await analyzeViaLocalServer();
    return;
  }

  await analyzeInBrowser();
}

async function analyzeViaLocalServer() {
  analyzeButton.disabled = true;
  resetSummary();
  try {
    const startedAt = performance.now();
    setStatus("로컬 추론 서버에 요청 중...", "loading");
    const response = await fetch(CONFIG.localServerUrl, {
      method: "POST",
      headers: {
        "Content-Type": state.file.type || "application/octet-stream",
      },
      body: state.file,
    });
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const message = payload?.message || `HTTP ${response.status}`;
      throw new Error(
        `로컬 추론 서버 호출에 실패했습니다. ${message}. \`python scripts/run_extension_inference_server.py --weights_path ckpt_best.pth\` 를 먼저 실행해 주세요.`,
      );
    }

    if (!payload) {
      throw new Error("로컬 추론 서버 응답을 해석하지 못했습니다.");
    }

    if (payload.status === "no_face") {
      drawOverlay(state.image, []);
      renderNoFaceResult(payload.num_detected_faces ?? 0);
      setStatus("얼굴을 찾지 못했습니다.", "done");
      timingChip.textContent = `${(payload.server_elapsed_ms || 0).toFixed(0)} ms`;
      return;
    }

    if (payload.status !== "ok") {
      throw new Error(
        payload.message || "로컬 추론 서버가 분석을 완료하지 못했습니다.",
      );
    }

    const normalizedResult = normalizeServerResult(payload);
    drawOverlay(state.image, normalizedResult.faces);
    renderResult(normalizedResult);
    const elapsed = performance.now() - startedAt;
    setStatus("분석 완료.", "done");
    timingChip.textContent = `${elapsed.toFixed(0)} ms`;
  } catch (error) {
    console.error(error);
    renderError(error);
    setStatus(error.message || "분석 중 오류가 발생했습니다.", "error");
  } finally {
    analyzeButton.disabled = !state.file;
  }
}

async function analyzeInBrowser() {
  analyzeButton.disabled = true;
  resetSummary();
  try {
    const startedAt = performance.now();
    setStatus("얼굴 탐지 모델 로딩 중...", "loading");
    const detectorResult = await runDetector(state.image);
    if (detectorResult.selectedDetections.length === 0) {
      drawOverlay(state.image, []);
      renderNoFaceResult(detectorResult.detections.length);
      setStatus("얼굴을 찾지 못했습니다.", "done");
      return;
    }

    setStatus("Classifier 모델 로딩 중...", "loading");
    const classifierSession = await getClassifierSession();
    setStatus("얼굴별 딥페이크 판별 중...", "running");
    const faceResults = [];

    for (
      let index = 0;
      index < detectorResult.selectedDetections.length;
      index += 1
    ) {
      const detection = detectorResult.selectedDetections[index];
      const cropCanvas = cropFaceToCanvas(
        state.image,
        detection.bbox,
        CONFIG.margin,
      );
      const classifierResult = await runClassifier(
        classifierSession,
        cropCanvas,
      );
      faceResults.push({
        faceIndex: index,
        bbox: detection.bbox.map((value) => Math.round(value)),
        detConfidence: detection.confidence,
        fakeProb: classifierResult.fakeProb,
        predLabelId: classifierResult.predLabelId,
        predLabel:
          classifierResult.predLabelId === 1 &&
          classifierResult.fakeProb >= CONFIG.fakeThreshold
            ? "fake"
            : "real",
        logits: classifierResult.logits,
        cropSize: [cropCanvas.width, cropCanvas.height],
      });
    }

    const aggregate = aggregateResults(
      faceResults,
      detectorResult.detections.length,
    );
    drawOverlay(state.image, faceResults);
    renderResult(aggregate);
    const elapsed = performance.now() - startedAt;
    setStatus("분석 완료.", "done");
    timingChip.textContent = `${elapsed.toFixed(0)} ms`;
  } catch (error) {
    console.error(error);
    renderError(error);
    setStatus(error.message || "분석 중 오류가 발생했습니다.", "error");
  } finally {
    analyzeButton.disabled = !state.file;
  }
}

function normalizeServerResult(result) {
  return {
    status: result.status,
    imageFake: result.image_fake,
    imageFakeProb: result.image_fake_prob,
    imagePredLabel: result.image_pred_label,
    numDetectedFaces: result.num_detected_faces,
    numFaces: result.num_faces,
    faces: (result.faces || []).map((face) => ({
      faceIndex: face.face_index,
      bbox: face.bbox,
      detConfidence: face.det_confidence,
      fakeProb: face.fake_prob,
      predLabelId: face.pred_label_id,
      predLabel: face.pred_label,
      logits: face.logits,
      cropSize: face.crop_size,
    })),
    summary: {
      maxFakeProb: result.summary?.max_fake_prob ?? result.image_fake_prob,
      meanFakeProb: result.summary?.mean_fake_prob ?? result.image_fake_prob,
      selectedFaceIndex: result.summary?.selected_face_index ?? 0,
      fakeThreshold: result.summary?.fake_threshold ?? CONFIG.fakeThreshold,
    },
  };
}

async function getClassifierSession() {
  if (state.classifierSession) {
    return state.classifierSession;
  }

  if (!state.classifierSessionPromise) {
    state.classifierSessionPromise = (async () => {
      const runtime = await ensureOrt();
      const executionProviders = [];

      if (supportsWebGpu()) {
        executionProviders.push("webgpu");
      }
      executionProviders.push("wasm");

      let lastError = null;
      for (const provider of executionProviders) {
        try {
          setStatus(
            `Classifier 모델 로딩 중... ${provider.toUpperCase()} 사용`,
            "loading",
          );
          const classifier = await runtime.InferenceSession.create(
            MODEL_URLS.classifier,
            {
              executionProviders: [provider],
              graphOptimizationLevel: "all",
            },
          );
          state.classifierSession = classifier;
          return classifier;
        } catch (error) {
          lastError = error;
        }
      }

      state.classifierSessionPromise = null;
      const message = `Classifier ONNX 세션을 생성하지 못했습니다. webgpu/wasm 모두 실패했습니다. ${lastError?.message || lastError}`;
      throw new Error(message);
    })();
  }

  return state.classifierSessionPromise;
}

function supportsWebGpu() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

async function ensureOrt() {
  if (ort) {
    return ort;
  }

  ort = await import("./vendor/ort.all.min.mjs");
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = "warning";
  return ort;
}

async function runDetector(image) {
  setStatus("얼굴 탐지 중...", "running");
  const runtime = await ensureOrt();
  const detections = await detectFacesWithOnnx(image, {
    runtime,
    letterboxImageToCanvas,
    tensorFromCanvas,
  });
  const selectedDetections = selectFaces(detections);
  return { detections, selectedDetections };
}

async function runClassifier(session, cropCanvas) {
  const runtime = await ensureOrt();
  const classifierSpec = detectClassifierSpec(session);
  const resizedCanvas = resizeCanvas(
    cropCanvas,
    classifierSpec.inputSize,
    classifierSpec.inputSize,
  );
  const imageInputFloat32 = tensorFromCanvas(resizedCanvas, {
    width: classifierSpec.inputSize,
    height: classifierSpec.inputSize,
    normalize: true,
    mean: classifierSpec.mean,
    std: classifierSpec.std,
  });
  const data = classifierSpec.useFloat16
    ? float32ArrayToFloat16Bits(imageInputFloat32)
    : imageInputFloat32;
  const feeds = {
    [classifierSpec.inputName]: new runtime.Tensor(
      classifierSpec.tensorType,
      data,
      [1, 3, classifierSpec.inputSize, classifierSpec.inputSize],
    ),
  };

  if (classifierSpec.includeBoundaryInput) {
    const ifBoundaryFloat32 = new Float32Array(256).fill(1.0);
    feeds.if_boundary = new runtime.Tensor(
      "float16",
      float32ArrayToFloat16Bits(ifBoundaryFloat32),
      [1, 256],
    );
  }

  const outputs = await session.run(feeds);
  return parseClassifierOutput(outputs);
}

function detectClassifierSpec(session) {
  const hasPixelValuesInput = session.inputNames.includes("pixel_values");
  const inputName = hasPixelValuesInput ? "pixel_values" : "image";
  const modelInputSize = getModelInputSpatialSize(session, inputName);

  return {
    inputName,
    inputSize: hasPixelValuesInput
      ? modelInputSize || 224
      : modelInputSize || CONFIG.classifierInputSize,
    mean: hasPixelValuesInput ? CONFIG.hfMean : CONFIG.mean,
    std: hasPixelValuesInput ? CONFIG.hfStd : CONFIG.std,
    tensorType: hasPixelValuesInput ? "float32" : "float16",
    includeBoundaryInput: !hasPixelValuesInput,
    useFloat16: !hasPixelValuesInput,
  };
}

function getModelInputSpatialSize(session, inputName) {
  const metadata = session.inputMetadata?.[inputName];
  if (!metadata || !Array.isArray(metadata.dims)) {
    return null;
  }

  const height = metadata.dims[2];
  const width = metadata.dims[3];
  if (
    Number.isInteger(height) &&
    Number.isInteger(width) &&
    height > 0 &&
    width > 0
  ) {
    return Math.min(height, width);
  }

  return null;
}

function parseClassifierOutput(outputs) {
  const logitsTensor =
    outputs.logits ||
    findTensor(outputs, (name, tensor) => {
      const dims = tensor?.dims || [];
      return (
        name !== "fake_prob" &&
        dims.length === 2 &&
        Number.isInteger(dims[dims.length - 1]) &&
        dims[dims.length - 1] > 1
      );
    });
  const fakeProbTensor = outputs.fake_prob;

  const logits = logitsTensor
    ? Array.from(readTensorAsFloat32(logitsTensor.data))
    : [];
  const pred = {
    logits,
    fakeProb: 0,
    predLabelId: 0,
  };

  if (fakeProbTensor) {
    pred.fakeProb = readTensorAsFloat32(fakeProbTensor.data)[0];
    pred.predLabelId =
      logits.length >= 2 && logits[1] > logits[0]
        ? 1
        : pred.fakeProb >= CONFIG.fakeThreshold
          ? 1
          : 0;
    return pred;
  }

  if (logits.length === 0) {
    throw new Error(
      "Classifier 출력에서 logits 또는 fake_prob를 찾지 못했습니다.",
    );
  }

  const probs = softmax(logits);
  pred.fakeProb = probs[HF_DEEPFAKE_CLASS_INDEX] ?? probs[1] ?? probs[0];
  pred.predLabelId = pred.fakeProb >= CONFIG.fakeThreshold ? 1 : 0;
  return pred;
}

function findTensor(outputs, predicate) {
  for (const [name, tensor] of Object.entries(outputs)) {
    if (predicate(name, tensor)) {
      return tensor;
    }
  }
  return null;
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const expValues = values.map((value) => Math.exp(value - maxValue));
  const total = expValues.reduce((acc, value) => acc + value, 0);
  return expValues.map((value) => value / total);
}

function selectFaces(detections) {
  const filtered = [];

  for (const detection of detections) {
    const confidence = Number(detection.confidence || 0);
    if (confidence < CONFIG.minConfidence) {
      continue;
    }

    const width = detection.bbox[2] - detection.bbox[0];
    const height = detection.bbox[3] - detection.bbox[1];
    if (width < CONFIG.minFaceSize || height < CONFIG.minFaceSize) {
      continue;
    }

    filtered.push({
      ...detection,
      area: width * height,
    });
  }

  filtered.sort((left, right) => right.confidence - left.confidence);

  if (CONFIG.topK == null) {
    return filtered;
  }

  if (CONFIG.topK <= 0) {
    return [];
  }

  return filtered.slice(0, CONFIG.topK);
}

function aggregateResults(faceResults, numDetectedFaces) {
  const fakeProbabilities = faceResults.map((face) => face.fakeProb);
  const maxFakeProb = Math.max(...fakeProbabilities);
  const selectedFaceIndex = fakeProbabilities.indexOf(maxFakeProb);
  const meanFakeProb =
    fakeProbabilities.reduce((sum, value) => sum + value, 0) /
    fakeProbabilities.length;
  const imagePredLabel = maxFakeProb >= CONFIG.fakeThreshold ? "fake" : "real";

  return {
    status: "ok",
    imageFake: imagePredLabel === "fake" ? 1 : 0,
    imageFakeProb: maxFakeProb,
    imagePredLabel,
    numDetectedFaces,
    numFaces: faceResults.length,
    faces: faceResults,
    summary: {
      maxFakeProb,
      meanFakeProb,
      selectedFaceIndex,
      fakeThreshold: CONFIG.fakeThreshold,
    },
  };
}

function renderResult(result) {
  summaryPanel.className = `summary-panel ${result.imagePredLabel}`;
  summaryPanel.innerHTML = `
    <span class="summary-badge">${result.imagePredLabel === "fake" ? "deepfake suspected" : "looks real"}</span>
    <p class="summary-title">${result.imagePredLabel === "fake" ? "딥페이크 가능성이 높습니다." : "실사로 분류되었습니다."}</p>
    <p class="summary-copy">최종 판정은 얼굴별 fake probability의 최댓값으로 집계합니다.</p>
    <div class="summary-metrics">
      <div class="metric-tile">
        <span class="metric-label">Image Fake Prob</span>
        <span class="metric-value">${formatProbability(result.imageFakeProb)}</span>
      </div>
      <div class="metric-tile">
        <span class="metric-label">Detected Faces</span>
        <span class="metric-value">${result.numFaces} / ${result.numDetectedFaces}</span>
      </div>
      <div class="metric-tile">
        <span class="metric-label">Mean Face Prob</span>
        <span class="metric-value">${formatProbability(result.summary.meanFakeProb)}</span>
      </div>
      <div class="metric-tile">
        <span class="metric-label">Decision Threshold</span>
        <span class="metric-value">${CONFIG.fakeThreshold.toFixed(2)}</span>
      </div>
    </div>
  `;

  faceList.innerHTML = result.faces
    .map(
      (face) => `
        <article class="face-card ${face.predLabel}">
          <div class="face-head">
            <div>
              <p class="face-name">Face ${face.faceIndex + 1}</p>
              <p class="face-meta">bbox ${face.bbox.join(", ")} | det ${face.detConfidence.toFixed(3)}</p>
            </div>
            <span class="face-score ${face.predLabel}">${formatProbability(face.fakeProb)}</span>
          </div>
          <p class="face-meta">판정: ${face.predLabel.toUpperCase()} | crop ${face.cropSize[0]}×${face.cropSize[1]}</p>
          <p class="face-logits">logits: [${face.logits.map((value) => value.toFixed(3)).join(", ")}]</p>
        </article>
      `,
    )
    .join("");
}

function renderNoFaceResult(numDetectedFaces) {
  summaryPanel.className = "summary-panel empty";
  summaryPanel.innerHTML = `
    <p class="summary-title">얼굴을 찾지 못했습니다.</p>
    <p class="summary-copy">검출된 얼굴 수: ${numDetectedFaces}. threshold 또는 이미지 품질을 확인해 주세요.</p>
  `;
  faceList.innerHTML = "";
}

function renderError(error) {
  summaryPanel.className = "summary-panel fake";
  summaryPanel.innerHTML = `
    <span class="summary-badge">error</span>
    <p class="summary-title">분석에 실패했습니다.</p>
    <p class="summary-copy">${escapeHtml(error.message || String(error))}</p>
  `;
  faceList.innerHTML = "";
}

function resetUi() {
  state.file = null;
  state.image = null;
  imageInput.value = "";
  analyzeButton.disabled = true;
  timingChip.textContent = "idle";
  previewContext.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewContext.fillStyle = "rgba(255,250,244,1)";
  previewContext.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  resetSummary();
  setStatus("브라우저에서 ONNX detector + classifier로 바로 분석합니다.", "idle");
}

function resetSummary() {
  summaryPanel.className = "summary-panel empty";
  summaryPanel.innerHTML = `
    <p class="summary-title">아직 분석 결과가 없습니다.</p>
    <p class="summary-copy">이미지를 추가하면 얼굴별 점수와 최종 이미지 판정이 표시됩니다.</p>
  `;
  faceList.innerHTML = "";
}

function drawPreviewImage(image) {
  const fitted = fitCanvas(
    previewCanvas,
    image.naturalWidth,
    image.naturalHeight,
  );
  previewContext.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewContext.fillStyle = "rgba(255,250,244,1)";
  previewContext.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewContext.drawImage(
    image,
    fitted.dx,
    fitted.dy,
    fitted.drawWidth,
    fitted.drawHeight,
  );
}

function drawOverlay(image, faceResults) {
  const fitted = fitCanvas(
    previewCanvas,
    image.naturalWidth,
    image.naturalHeight,
  );
  previewContext.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewContext.fillStyle = "rgba(255,250,244,1)";
  previewContext.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewContext.drawImage(
    image,
    fitted.dx,
    fitted.dy,
    fitted.drawWidth,
    fitted.drawHeight,
  );

  previewContext.lineWidth = 3;
  previewContext.font = '13px "Avenir Next", "Segoe UI", sans-serif';

  for (const face of faceResults) {
    const [x1, y1, x2, y2] = face.bbox;
    const color = face.predLabel === "fake" ? "#b53d34" : "#0e7b58";
    const drawX = fitted.dx + x1 * fitted.scale;
    const drawY = fitted.dy + y1 * fitted.scale;
    const drawWidth = (x2 - x1) * fitted.scale;
    const drawHeight = (y2 - y1) * fitted.scale;

    previewContext.strokeStyle = color;
    previewContext.fillStyle = color;
    previewContext.strokeRect(drawX, drawY, drawWidth, drawHeight);
    const label = `${face.faceIndex + 1} · ${face.predLabel} · ${formatProbability(face.fakeProb)}`;
    const textWidth = previewContext.measureText(label).width + 12;
    const textY = Math.max(18, drawY - 10);
    previewContext.fillRect(drawX, textY - 16, textWidth, 18);
    previewContext.fillStyle = "#fff9f5";
    previewContext.fillText(label, drawX + 6, textY - 3);
  }
}

function readTensorAsFloat32(data) {
  if (data instanceof Float32Array) {
    return data;
  }

  if (data instanceof Uint16Array) {
    const output = new Float32Array(data.length);
    for (let index = 0; index < data.length; index += 1) {
      output[index] = float16BitsToNumber(data[index]);
    }
    return output;
  }

  return Float32Array.from(data);
}

function float32ArrayToFloat16Bits(values) {
  const output = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = numberToFloat16Bits(values[index]);
  }
  return output;
}

function numberToFloat16Bits(value) {
  const floatView = new Float32Array(1);
  const intView = new Uint32Array(floatView.buffer);
  floatView[0] = value;
  const bits = intView[0];
  const sign = (bits >>> 16) & 0x8000;
  const mantissa = bits & 0x007fffff;
  const exponent = (bits >>> 23) & 0xff;

  if (exponent === 0xff) {
    if (mantissa !== 0) {
      return sign | 0x7e00;
    }
    return sign | 0x7c00;
  }

  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) {
    return sign | 0x7c00;
  }

  if (halfExponent <= 0) {
    if (halfExponent < -10) {
      return sign;
    }

    const subnormal = (mantissa | 0x00800000) >> (1 - halfExponent);
    return sign | ((subnormal + 0x00001000) >> 13);
  }

  return sign | (halfExponent << 10) | ((mantissa + 0x00001000) >> 13);
}

function float16BitsToNumber(value) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x03ff;
  let bits = 0;

  if (exponent === 0) {
    if (mantissa === 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      mantissa &= 0x03ff;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }

  const intView = new Uint32Array(1);
  const floatView = new Float32Array(intView.buffer);
  intView[0] = bits;
  return floatView[0];
}

function formatProbability(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setStatus(message, mode) {
  statusText.textContent = message;
  timingChip.textContent = mode;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("이미지를 읽지 못했습니다."));
    };
    image.src = objectUrl;
  });
}
