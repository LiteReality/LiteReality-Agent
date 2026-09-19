"""Scan photographs for merge/layout review, independent of generated object crops."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .shell import object_footprint


def prepare(scan, scene_dir, shell):
    """Project current boxes onto raw frames; retain source poses beside the photographs."""
    scene_dir = Path(scene_dir)
    capture = scene_dir.parent.parent / 'rgbd' / scan
    images = sorted((capture / 'image').glob('frame_*.jpg'))
    if not images:
        return {"images": 0, "reason": "no capture frames"}
    from ..ingest.preprocessing.vendor.litereality.object_image_extraction import project_bbox_2d

    class Frames:
        def __init__(self):
            self.ks, self.poses = {}, {}

        def intrinsic(self, index):
            if index not in self.ks:
                self.ks[index] = np.load(capture / 'intrinsic' / f'intrinsic_{index}.npy')
            return self.ks[index]

        def pose_inv(self, index):
            if index not in self.poses:
                self.poses[index] = np.linalg.inv(np.load(
                    capture / 'extrinsic' / f'extrinsic_{index}.npy'))
            return self.poses[index]

    frames, count = Frames(), 0
    capture_key = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                   for folder in ('image', 'intrinsic', 'extrinsic')
                   for p in sorted((capture / folder).glob('*')) if p.is_file()]
    for oid, obj in shell['objects'].items():
        folder = scene_dir / 'references' / oid
        stamp = folder / 'geometry.json'
        digest = hashlib.sha256(json.dumps([obj, capture_key], sort_keys=True).encode()).hexdigest()
        if stamp.is_file():
            try:
                prior = json.loads(stamp.read_text())
            except (OSError, ValueError):
                prior = {}
            if prior.get('fingerprint') == digest and all(
                    (folder / name).is_file() for name in prior.get('images', [])):
                count += len(prior.get('images', []))
                continue
        z0 = obj['center'][2] - obj['size'][2] / 2
        z1 = obj['center'][2] + obj['size'][2] / 2
        points = np.array([[x, z, -y] for z in (z0, z1) for x, y in object_footprint(obj)])
        ranked = []
        for path in images:
            index = int(path.stem.split('_')[-1])
            rect = project_bbox_2d(scan, points, index, frames=frames)
            if rect:
                x0, y0, x1, y1 = rect
                area = (x1 - x0) * (y1 - y0) / (256 * 192)
                if 0.01 < area < 0.85:
                    ranked.append((area, index, path, rect))
        ranked.sort(reverse=True)
        selected = []
        for item in ranked:
            if all(abs(item[1] - old[1]) >= 3 for old in selected):
                selected.append(item)
            if len(selected) == 2:
                break
        folder.mkdir(parents=True, exist_ok=True)
        # Only replace these generated review images, never user-supplied images elsewhere.
        for old in folder.glob('rank*.jpg'):
            old.unlink()
        names = []
        for rank, (_, _, path, rect) in enumerate(selected):
            with Image.open(path) as raw:
                picture = raw.convert('RGB')
            xy = [rect[0]*picture.width/256, rect[1]*picture.height/192,
                  rect[2]*picture.width/256, rect[3]*picture.height/192]
            draw = ImageDraw.Draw(picture)
            draw.rectangle(xy, outline='red', width=5)
            draw.text(xy[:2], oid, fill='red')
            name = f'rank{rank}.jpg'
            picture.rotate(-90, expand=True).save(folder / name)
            names.append(name)
        stamp.write_text(json.dumps({'fingerprint': digest, 'object': obj, 'images': names}, indent=2))
        count += len(names)
    return {'images': count}


def review_merge(shell, members, scene_dir, model='sonnet'):
    """Conservative visual decision: only explicit, confident approval joins objects."""
    import subprocess

    refs = {oid: [str(p) for p in sorted(
        (Path(scene_dir) / 'references' / oid).glob('rank*.jpg'))[:2]] for oid in members}
    if not all(refs.values()):
        return {'action': 'separate', 'why': 'missing member photographs'}
    context = {oid: shell['objects'][oid] for oid in members}
    prompt = ('Read ALL listed scan images with the Read tool. Decide whether these detected '
              'boxes describe ONE physical object or one continuous installed cabinet run. '
              'A hanging shelf and floor cabinet, stacked separate units, or units enclosing '
              'empty space occupied by a desk must remain separate even if their boxes touch. '
              'Use the elevations and photographs. Uncertainty means separate. Reply ONLY JSON: '
              '{"action":"merge"|"separate","confidence":0.0,"why":"visual evidence"}.\n'
              + json.dumps({'objects': context, 'reference_images': refs}))
    try:
        done = subprocess.run(['claude', '-p', prompt, '--output-format', 'text', '--model', model,
                               '--allowed-tools', 'Read'], capture_output=True, text=True, timeout=180)
        text = done.stdout.strip()
        decision = json.loads(text[text.find('{'):text.rfind('}')+1])
        confidence = float(decision.get('confidence', 0))
        if (done.returncode or decision.get('action') != 'merge'
                or not math.isfinite(confidence) or not 0.8 <= confidence <= 1):
            decision['action'] = 'separate'
        return decision
    except (OSError, ValueError, TypeError, AttributeError, subprocess.TimeoutExpired) as exc:
        return {'action': 'separate', 'why': f'review unavailable: {type(exc).__name__}'}
