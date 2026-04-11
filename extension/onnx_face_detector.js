import { clampBbox } from "./image_utils.js";

const DETECTOR_MODEL_URL = chrome.runtime.getURL("models/face_detector.onnx");
const DETECTOR_INPUT_SIZE = 640;
const DETECTOR_FILL = [114, 114, 114];

let detectorSessionPromise = null;

export async function detectFacesWithOnnx(image, options = {}) {
  const runtime = options.runtime;
  const letterbox = options.letterboxImageToCanvas;
  const tensorFromCanvas = options.tensorFromCanvas;

  if (!runtime) {
    throw new Error("ONNX Runtime Web instance is required for face detection.");
  }

  if (typeof letterbox !== "function" || typeof tensorFromCanvas !== "function") {
    throw new Error("Face detector preprocessing helpers are required.");
  }

  const session = await getDetectorSession(runtime);
  const inputName = session.inputNames[0];
  const { canvas, metadata } = letterbox(image, DETECTOR_INPUT_SIZE, DETECTOR_FILL);
  const input = tensorFromCanvas(canvas, {
    width: DETECTOR_INPUT_SIZE,
    height: DETECTOR_INPUT_SIZE,
    normalize: false,
  });
  const feeds = {
    [inputName]: new runtime.Tensor("float32", input, [1, 3, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE]),
  };
  const outputs = await session.run(feeds);
  const outputName = session.outputNames[0] || Object.keys(outputs)[0];
  const rawDetections = parseDetectorOutput(outputs[outputName]);
  return scaleDetectionsBack(rawDetections, metadata).sort(
    (left, right) => right.confidence - left.confidence,
  );
}

async function getDetectorSession(runtime) {
  if (!detectorSessionPromise) {
    detectorSessionPromise = runtime.InferenceSession.create(DETECTOR_MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }).catch((error) => {
      detectorSessionPromise = null;
      throw error;
    });
  }

  return detectorSessionPromise;
}

function parseDetectorOutput(outputValue) {
  let data = outputValue?.data ?? outputValue;
  let dims = outputValue?.dims || [];

  if (!(data instanceof Float32Array)) {
    data = Float32Array.from(data);
  }

  if (dims.length === 0 && Array.isArray(outputValue)) {
    dims = inferArrayShape(outputValue);
    data = flattenArray(outputValue);
  }

  if (dims.length === 3 && dims[0] === 1) {
    dims = [dims[1], dims[2]];
  } else if (dims.length === 3 && dims[2] === 1) {
    dims = [dims[0], dims[1]];
  }

  if (dims.length !== 2) {
    throw new Error(`Unsupported detector output shape: ${JSON.stringify(dims)}`);
  }

  let rows = dims[0];
  let columns = dims[1];
  if (rows >= 6 && columns < 6) {
    [rows, columns] = [columns, rows];
  }

  if (columns < 6) {
    throw new Error(`Detector output must expose at least 6 columns, got ${rows}x${columns}`);
  }

  const detections = [];
  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    const row = readRow(data, dims, rowIndex, rows, columns);
    if (row.slice(0, 6).some((value) => !Number.isFinite(value))) {
      continue;
    }

    const confidence = Number(row[4]);
    if (confidence <= 0) {
      continue;
    }

    detections.push({
      bbox: [Number(row[0]), Number(row[1]), Number(row[2]), Number(row[3])],
      confidence,
      classId: Number(row[5]),
    });
  }

  return detections;
}

function readRow(data, originalDims, rowIndex, rows, columns) {
  const transposed = originalDims[0] !== rows;
  const row = new Array(columns);

  for (let columnIndex = 0; columnIndex < columns; columnIndex += 1) {
    const dataIndex = transposed
      ? columnIndex * rows + rowIndex
      : rowIndex * columns + columnIndex;
    row[columnIndex] = data[dataIndex];
  }

  return row;
}

function scaleDetectionsBack(detections, metadata) {
  const scaled = [];

  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.bbox;
    const scaledBbox = clampBbox(
      [
        (x1 - metadata.padX) / metadata.scale,
        (y1 - metadata.padY) / metadata.scale,
        (x2 - metadata.padX) / metadata.scale,
        (y2 - metadata.padY) / metadata.scale,
      ],
      metadata.origWidth,
      metadata.origHeight,
    ).map((value) => Math.round(value));

    scaled.push({
      bbox: scaledBbox,
      confidence: detection.confidence,
      landmarks: null,
    });
  }

  return scaled;
}

function inferArrayShape(value) {
  const shape = [];
  let current = value;

  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }

  return shape;
}

function flattenArray(value) {
  return Float32Array.from(value.flat(Infinity));
}
