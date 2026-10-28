"""Zero-shot object verification with CLIP: is this candidate box a phone or a calculator?

COCO has no calculator class, and both YOLOv8n and YOLO-World call calculators "cell phone"
with high confidence (Test_1: 0.61-0.66 on calculators). An open-vocabulary detector is
good at FINDING small hand-held objects (it found the back-row phone that YOLOv8n missed),
but not at telling these two apart. CLIP compares a crop of the box with short text
descriptions and was correct on all checked Test_1 cases (calculators 0.98-1.00; the
back-row phone held in a dark case: smartphone 0.69; a calculator lying in the same place
15 s later: calculator 0.98).

Crops are classified in one batch per frame and cached: a box that overlaps a box classified
in the last `cache_s` seconds (objects on desks hardly move) reuses that result.
"""

import numpy as np

LABELS = ["a smartphone", "a calculator", "a pencil case", "a pen", "a sheet of paper", "a hand"]


def decide(probs, phone_min=0.35):
    """CLIP probabilities of one candidate box -> ("cell phone" | "calculator", confidence) or None
    (pencil case, pen, hand: not an exam-relevant object)."""
    label = max(probs, key=probs.get)
    if label == "a smartphone" and probs[label] >= phone_min:
        return "cell phone", probs[label]
    if label == "a calculator":
        return "calculator", probs[label]
    return None


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class ClipVerifier:
    def __init__(self, model_name="ViT-B/32", margin=0.25, cache_s=2.0, device="cpu"):
        import clip
        import torch
        self.torch, self.clip = torch, clip
        self.model, self.pre = clip.load(model_name, device=device)
        self.model.eval()
        with torch.no_grad():
            tf = self.model.encode_text(clip.tokenize([f"a photo of {l}" for l in LABELS]))
        self.text = tf / tf.norm(dim=-1, keepdim=True)
        self.margin, self.cache_s = margin, cache_s
        self.cache = []                     # (t, box, label, probs)

    def _crop(self, frame, b):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = b
        pw, ph = x2 - x1, max(y2 - y1, 0.6 * (x2 - x1))      # thin boxes: keep some height
        cy = (y1 + y2) / 2
        y1, y2 = cy - ph / 2, cy + ph / 2
        m = self.margin
        return frame[int(max(0, y1 - m * ph)):int(min(H, y2 + m * ph)), int(max(0, x1 - m * pw)):int(min(W, x2 + m * pw))]

    def classify(self, frame, boxes, t):
        """-> [(label, {label: prob})] per box; label is one of LABELS."""
        import cv2
        from PIL import Image
        self.cache = [c for c in self.cache if t - c[0] <= self.cache_s]
        out, todo = [None] * len(boxes), []
        for i, b in enumerate(boxes):
            hit = next((c for c in self.cache if _iou(c[1], b) > 0.5), None)
            if hit:
                out[i] = (hit[2], hit[3])
            else:
                todo.append(i)
        if todo:
            ims = []
            for i in todo:
                crop = self._crop(frame, boxes[i])
                ims.append(self.pre(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
            with self.torch.no_grad():
                f = self.model.encode_image(self.torch.stack(ims))
                f = f / f.norm(dim=-1, keepdim=True)
                p = (100 * f @ self.text.T).softmax(-1).cpu().numpy()
            for i, row in zip(todo, p):
                probs = {l: round(float(v), 3) for l, v in zip(LABELS, row)}
                label = LABELS[int(np.argmax(row))]
                out[i] = (label, probs)
                self.cache.append((t, boxes[i], label, probs))
        return out
