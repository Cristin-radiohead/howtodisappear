import cv2
import numpy as np
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from ais_bench.infer.interface import InferSession

print("Loading model...")
session = InferSession(device_id=0, model_path="/root/yolo11n.om")
print("Model loaded")

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball",
    33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def postprocess(output, conf_thres=0.45, iou_thres=0.45):
    pred = output[0].T  # (8400, 84)
    cxcy = pred[:, :2]
    wh = pred[:, 2:4]
    boxes = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=1)

    # 直接取类别分数，不做 sigmoid（模型输出已含 sigmoid）
    class_scores = pred[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    mask = max_scores >= conf_thres
    if not np.any(mask):
        return []

    final_boxes = boxes[mask]
    final_scores = max_scores[mask]
    final_ids = class_ids[mask]

    w = final_boxes[:, 2] - final_boxes[:, 0]
    h = final_boxes[:, 3] - final_boxes[:, 1]
    valid = (w > 1) & (h > 1) & \
            (final_boxes[:, 0] >= 0) & (final_boxes[:, 1] >= 0) & \
            (final_boxes[:, 2] <= 640) & (final_boxes[:, 3] <= 640)

    if not np.any(valid):
        return []

    final_boxes = final_boxes[valid]
    final_scores = final_scores[valid]
    final_ids = final_ids[valid]

    indices = cv2.dnn.NMSBoxes(
        final_boxes.tolist(),
        final_scores.tolist(),
        conf_thres, iou_thres
    )

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                'class_id': int(final_ids[i]),
                'score': float(final_scores[i]),
                'box': final_boxes[i].tolist()
            })
    return results


def detect(img):
    img_lb, ratio, (dw, dh) = letterbox(img, (640, 640))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1))[np.newaxis, :])
    outputs = session.infer([tensor])
    dets = postprocess(outputs[0])

    h, w = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d['box']
        d['box'] = [
            max(0, min(w, (x1 - dw) / ratio)),
            max(0, min(h, (y1 - dh) / ratio)),
            max(0, min(w, (x2 - dw) / ratio)),
            max(0, min(h, (y2 - dh) / ratio)),
        ]
    return dets


def parse_multipart(body, boundary):
    sep = b'\r\n\r\n'
    start = body.find(sep)
    if start == -1:
        return None
    start += len(sep)
    end_marker = b'\r\n--' + boundary
    end = body.find(end_marker, start)
    if end == -1:
        end = len(body)
    return body[start:end]


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>YOLO Image Detection</title>
<style>
body { font-family: sans-serif; background: #1a1a1a; color: #eee; text-align: center; padding: 20px; }
h1 { font-weight: 400; }
input[type=file] { padding: 10px; background: #333; color: #eee; border-radius: 6px; }
button { padding: 10px 24px; background: #4ade80; color: #111; border: none; border-radius: 6px; font-size: 15px; cursor: pointer; margin: 10px; }
button:hover { background: #22c55e; }
#result img { max-width: 90vw; border: 2px solid #444; border-radius: 8px; margin-top: 16px; }
#stats { color: #4ade80; font-size: 15px; margin: 12px; }
#loading { color: #fbbf24; }
</style>
</head>
<body>
<h1>YOLO Image Detection</h1>
<p>Select an image, click Detect.</p >
<input type="file" id="fileInput" accept="image/*">
<br>
<button onclick="runDetect()">Detect</button>
<div id="stats"></div>
<div id="loading"></div>
<div id="result"></div>
<script>
function runDetect() {
    var fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) { alert('Please select an image'); return; }
    var file = fileInput.files[0];
    document.getElementById('loading').textContent = 'Detecting, please wait...';
    document.getElementById('stats').textContent = '';
    document.getElementById('result').innerHTML = '';
    var formData = new FormData();
    formData.append('file', file);
    fetch('/detect', { method: 'POST', body: formData })
        .then(function(res) { return res.json(); })
        .then(function(data) {
            document.getElementById('stats').textContent =
                'Detected ' + data.count + ' objects: ' + data.names.join(', ');
            var img = document.createElement('img');
            img.src = '/result.jpg?t=' + Date.now();
            document.getElementById('result').appendChild(img);
            document.getElementById('loading').textContent = '';
        })
        .catch(function(e) {
            document.getElementById('stats').textContent = 'Error: ' + e;
            document.getElementById('loading').textContent = '';
        });
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path.startswith('/result.jpg'):
            try:
                with open('/root/latest_result.jpg', 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/detect':
            self.send_response(404)
            self.end_headers()
            return

        content_type = self.headers.get('Content-Type', '')
        if 'boundary=' not in content_type:
            self.send_response(400)
            self.end_headers()
            return
        boundary = content_type.split('boundary=')[1].strip().encode()
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        img_bytes = parse_multipart(body, boundary)
        if not img_bytes:
            self.send_response(400)
            self.end_headers()
            return

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            self.send_response(400)
            self.end_headers()
            return

        dets = detect(img)

        names = []
        for d in dets:
            name = COCO_NAMES.get(d['class_id'], 'cls%d' % d['class_id'])
            names.append(name)
            x1, y1, x2, y2 = [int(round(v)) for v in d['box']]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, '%s %.2f' % (name, d['score']),
                        (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print("Detected:", list(zip(names, [round(d['score'], 2) for d in dets])))

        cv2.imwrite('/root/latest_result.jpg', img)

        result = {"count": len(dets), "names": names}
        data = json.dumps(result).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    print("Server started: http://0.0.0.0:5000")
    server = HTTPServer(('0.0.0.0', 5000), Handler)
    server.serve_forever()
