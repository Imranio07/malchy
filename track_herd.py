"""Malchy — track a herd in video frames: SAM 3 detects animals on every frame,
a lightweight tracker links detections into IDs (memory stays flat on 8 GB)."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().tolist()
    return list(x)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def iou(a, b):
    inter = inter_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def shift(b, dx, dy):
    return [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]


def dedup(boxes, scores, iou_thr=0.5, inside_thr=0.8):
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept = []
    for i in order:
        bi, ai = boxes[i], area(boxes[i])
        if any(iou(bi, boxes[j]) > iou_thr or (ai > 0 and inter_area(bi, boxes[j]) / ai > inside_thr)
               for j in kept):
            continue
        kept.append(i)
    return kept


def detect(processor, image, prompt, threshold, iou_thr):
    width, height = image.size
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        out = processor.set_text_prompt(state=state, prompt=prompt)
    boxes, scores = to_list(out["boxes"]), to_list(out["scores"])
    if boxes and max(max(b) for b in boxes) <= 1.0:
        boxes = [[b[0] * width, b[1] * height, b[2] * width, b[3] * height] for b in boxes]
    confident = [i for i, s in enumerate(scores) if s >= threshold]
    kept = dedup([boxes[i] for i in confident], [scores[i] for i in confident], iou_thr)
    return [boxes[confident[k]] for k in kept]


class HerdTracker:
    def __init__(self, diag, match_iou=0.2, max_jump=0.06, max_missing=6):
        self.diag = diag
        self.match_iou = match_iou
        self.max_jump = max_jump * diag
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 1
        self.camera = {}
        self._cum = (0.0, 0.0)

    def update(self, frame, boxes):
        active = [t for t, tr in self.tracks.items() if frame - tr["last"] <= self.max_missing]
        dx = dy = 0.0
        if active and boxes:
            prev = [centre(self.tracks[t]["box"]) for t in active]
            offsets = []
            for b in boxes:
                c = centre(b)
                p = min(prev, key=lambda q: (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2)
                offsets.append((c[0] - p[0], c[1] - p[1]))
            dx = float(np.median([o[0] for o in offsets]))
            dy = float(np.median([o[1] for o in offsets]))
        self._cum = (self._cum[0] + dx, self._cum[1] + dy)
        self.camera[frame] = self._cum

        pairs = []
        for di, b in enumerate(boxes):
            cb = centre(b)
            for t in active:
                predicted = shift(self.tracks[t]["box"], dx, dy)
                cp = centre(predicted)
                dist = math.hypot(cb[0] - cp[0], cb[1] - cp[1])
                overlap = iou(b, predicted)
                if overlap >= self.match_iou or dist <= self.max_jump:
                    pairs.append((overlap - dist / self.diag, di, t))
        pairs.sort(reverse=True)
        assigned, used_tracks = {}, set()
        for _, di, t in pairs:
            if di in assigned or t in used_tracks:
                continue
            assigned[di] = t
            used_tracks.add(t)

        visible = {}
        for di, b in enumerate(boxes):
            t = assigned.get(di)
            if t is None:
                t = self.next_id
                self.next_id += 1
                self.tracks[t] = {"hits": 0, "history": {}}
            tr = self.tracks[t]
            tr["box"], tr["last"] = b, frame
            tr["hits"] += 1
            tr["history"][frame] = b
            visible[t] = b
        return visible

    def ground(self, frame, b):
        cx, cy = centre(b)
        sx, sy = self.camera.get(frame, (0.0, 0.0))
        return cx - sx, cy - sy


def analyse(tracker, confirmed, last_frame, stray_factor, still_fraction, min_frames):
    recent = {}
    for t in confirmed:
        hist = tracker.tracks[t]["history"]
        f = max(hist)
        if last_frame - f <= tracker.max_missing:
            recent[t] = centre(hist[f])
    strays = []
    if len(recent) >= 3:
        cx = float(np.median([c[0] for c in recent.values()]))
        cy = float(np.median([c[1] for c in recent.values()]))
        dists = {t: math.hypot(c[0] - cx, c[1] - cy) for t, c in recent.items()}
        typical = float(np.median(list(dists.values()))) or 1.0
        strays = [t for t, d in dists.items() if d > stray_factor * typical]
    motionless = []
    for t in confirmed:
        hist = tracker.tracks[t]["history"]
        if len(hist) < min_frames:
            continue
        pts = [tracker.ground(f, hist[f]) for f in sorted(hist)]
        span = max(math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) for p in pts)
        if span < still_fraction * tracker.diag:
            motionless.append(t)
    return sorted(strays), sorted(motionless)


def main():
    parser = argparse.ArgumentParser(description="Track a herd in video frames with SAM 3")
    parser.add_argument("frames")
    parser.add_argument("--prompt", default="sheep")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--stray-factor", type=float, default=2.5)
    parser.add_argument("--still-fraction", type=float, default=0.02)
    parser.add_argument("--min-frames", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Run `conda activate sam3` first.")
    paths = sorted(Path(args.frames).glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No .jpg frames in {args.frames}")

    h, w = cv2.imread(str(paths[0])).shape[:2]
    tracker = HerdTracker(diag=math.hypot(w, h))
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    per_frame = {}
    for f, path in enumerate(paths):
        boxes = detect(processor, Image.open(path).convert("RGB"), args.prompt, args.threshold, args.iou)
        per_frame[f] = tracker.update(f, boxes)
        print(f"frame {f + 1}/{len(paths)}: {len(boxes)} detected, {tracker.next_id - 1} tracks so far")

    confirmed = {t for t, tr in tracker.tracks.items() if tr["hits"] >= args.min_hits}
    if not confirmed:
        raise SystemExit(f"No '{args.prompt}' tracked. Try a lower --threshold or another clip.")
    strays, motionless = analyse(tracker, confirmed, len(paths) - 1,
                                 args.stray_factor, args.still_fraction, args.min_frames)

    first_seen = {t: min(tracker.tracks[t]["history"]) for t in confirmed}
    report = {
        "prompt": args.prompt,
        "method": "SAM 3 detection on every frame + IoU tracking with camera-motion compensation",
        "frames": len(paths),
        "unique_animals": len(confirmed),
        "max_visible_in_one_frame": max(sum(1 for t in v if t in confirmed) for v in per_frame.values()),
        "stray_ids": strays,
        "motionless_ids": motionless,
    }
    Path("herd_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    writer = cv2.VideoWriter("herd_tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f, path in enumerate(paths):
        img = cv2.imread(str(path))
        shown = {t: b for t, b in per_frame[f].items() if t in confirmed}
        for t, b in shown.items():
            colour = (0, 0, 255) if t in strays else (0, 220, 255) if t in motionless else (0, 200, 0)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(img, str(t), (x1 + 3, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        counted = sum(1 for t in confirmed if first_seen[t] <= f)
        label = f"{args.prompt}: {counted} counted | in frame: {len(shown)}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(img)
    writer.release()
    print("Saved herd_tracked.mp4 and herd_report.json")


if __name__ == "__main__":
    main()
