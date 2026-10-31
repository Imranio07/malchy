"""Malchy — count animals in one image with SAM 3, removing duplicate boxes."""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().tolist()
    return list(x)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def dedup(boxes, scores, iou_thr=0.6, inside_thr=0.8):
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept = []
    for i in order:
        bi, ai = boxes[i], area(boxes[i])
        duplicate = False
        for j in kept:
            bj = boxes[j]
            iw = max(0.0, min(bi[2], bj[2]) - max(bi[0], bj[0]))
            ih = max(0.0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
            inter = iw * ih
            union = ai + area(bj) - inter
            if union > 0 and inter / union > iou_thr:
                duplicate = True
                break
            if ai > 0 and inter / ai > inside_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return kept


def main():
    parser = argparse.ArgumentParser(description="Count animals in an image with SAM 3")
    parser.add_argument("image")
    parser.add_argument("--prompt", default="sheep")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Run `conda activate sam3` first.")

    image = Image.open(args.image).convert("RGB")
    width, height = image.size

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)

    boxes = to_list(output["boxes"])
    scores = to_list(output["scores"])
    if boxes and max(max(b) for b in boxes) <= 1.0:
        boxes = [[b[0] * width, b[1] * height, b[2] * width, b[3] * height] for b in boxes]

    confident = [i for i, s in enumerate(scores) if s >= args.threshold]
    kept = dedup([boxes[i] for i in confident], [scores[i] for i in confident], args.iou)
    found = [(boxes[confident[k]], scores[confident[k]]) for k in kept]

    print(f"Raw detections: {len(confident)}, after removing duplicates: {len(found)}")

    draw = ImageDraw.Draw(image)
    for i, (box, score) in enumerate(found, start=1):
        draw.rectangle(box, outline=(255, 64, 64), width=3)
        draw.text((box[0] + 4, box[1] + 2), f"{i} ({score:.2f})", fill=(255, 255, 0))
    draw.text((10, 10), f"{args.prompt}: {len(found)}", fill=(255, 255, 0))

    out_path = Path(args.image).with_name(Path(args.image).stem + "_counted.jpg")
    image.save(out_path, quality=95)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
