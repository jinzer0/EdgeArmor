import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.deepfake_pipeline import DeepfakeDetectionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local inference server for the Chrome extension.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--config_path", default="ForensicsAdapter/config/test.yaml", help="Path to YAML config")
    parser.add_argument("--weights_path", default="ckpt_best.pth", help="Path to checkpoint weights")
    parser.add_argument("--device", default=None, help="Optional torch device override")
    return parser.parse_args()


def make_json_response(handler, status_code, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def build_handler(pipeline):
    class ExtensionInferenceHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            make_json_response(self, 200, {"status": "ok"})

        def do_GET(self):
            if self.path != "/health":
                make_json_response(self, 404, {"status": "error", "message": "Not found"})
                return

            make_json_response(
                self,
                200,
                {
                    "status": "ok",
                    "service": "edgearmor-extension-inference",
                },
            )

        def do_POST(self):
            if self.path != "/analyze":
                make_json_response(self, 404, {"status": "error", "message": "Not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0

            if content_length <= 0:
                make_json_response(self, 400, {"status": "error", "message": "Empty request body"})
                return

            body = self.rfile.read(content_length)
            started_at = time.perf_counter()

            try:
                image = Image.open(BytesIO(body)).convert("RGB")
            except UnidentifiedImageError:
                make_json_response(self, 400, {"status": "error", "message": "Unsupported image payload"})
                return
            except Exception as error:
                make_json_response(self, 400, {"status": "error", "message": str(error)})
                return

            try:
                result = pipeline.predict(image)
            except Exception as error:
                make_json_response(
                    self,
                    500,
                    {
                        "status": "error",
                        "message": f"Inference failed: {error}",
                    },
                )
                return

            result["server_elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            make_json_response(self, 200, result)

        def log_message(self, format_string, *args):
            return

    return ExtensionInferenceHandler


def main():
    args = parse_args()
    pipeline = DeepfakeDetectionPipeline(
        config_path=args.config_path,
        weights_path=args.weights_path,
        device=args.device,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(pipeline))
    print(f"EdgeArmor extension inference server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
