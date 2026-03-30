export function fitCanvas(canvas, sourceWidth, sourceHeight) {
  const scale = Math.min(canvas.width / sourceWidth, canvas.height / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  return {
    scale,
    drawWidth,
    drawHeight,
    dx: (canvas.width - drawWidth) / 2,
    dy: (canvas.height - drawHeight) / 2,
  };
}

export function clampBbox(bbox, imageWidth, imageHeight) {
  const x1 = Math.max(0, Math.min(imageWidth, bbox[0]));
  const y1 = Math.max(0, Math.min(imageHeight, bbox[1]));
  const x2 = Math.max(0, Math.min(imageWidth, bbox[2]));
  const y2 = Math.max(0, Math.min(imageHeight, bbox[3]));
  const left = Math.max(0, Math.min(x1, x2 - 1));
  const top = Math.max(0, Math.min(y1, y2 - 1));
  const right = Math.min(imageWidth, Math.max(x2, left + 1));
  const bottom = Math.min(imageHeight, Math.max(y2, top + 1));

  return [left, top, right, bottom];
}

export function expandBbox(bbox, margin, imageWidth, imageHeight) {
  const [x1, y1, x2, y2] = bbox;
  const bboxWidth = x2 - x1;
  const bboxHeight = y2 - y1;
  const marginX = bboxWidth * margin;
  const marginY = bboxHeight * margin;
  return clampBbox(
    [Math.floor(x1 - marginX), Math.floor(y1 - marginY), Math.ceil(x2 + marginX), Math.ceil(y2 + marginY)],
    imageWidth,
    imageHeight,
  );
}

export function cropFaceToCanvas(image, bbox, margin) {
  const [x1, y1, x2, y2] = expandBbox(bbox, margin, image.naturalWidth, image.naturalHeight);
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(image, x1, y1, width, height, 0, 0, width, height);
  return canvas;
}

export function resizeCanvas(sourceCanvas, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(sourceCanvas, 0, 0, width, height);
  return canvas;
}

export function tensorFromCanvas(canvas, options) {
  const context = canvas.getContext("2d");
  const imageData = context.getImageData(0, 0, options.width, options.height).data;
  const planeSize = options.width * options.height;
  const output = new Float32Array(planeSize * 3);

  for (let index = 0; index < planeSize; index += 1) {
    const pixelOffset = index * 4;
    const red = imageData[pixelOffset] / 255;
    const green = imageData[pixelOffset + 1] / 255;
    const blue = imageData[pixelOffset + 2] / 255;

    if (options.normalize) {
      output[index] = (red - options.mean[0]) / options.std[0];
      output[planeSize + index] = (green - options.mean[1]) / options.std[1];
      output[planeSize * 2 + index] = (blue - options.mean[2]) / options.std[2];
    } else {
      output[index] = red;
      output[planeSize + index] = green;
      output[planeSize * 2 + index] = blue;
    }
  }

  return output;
}
