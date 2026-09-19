"""Bind generated references to scan evidence, independent of repaired placement."""

import hashlib
import json

from ... import paths as config
from ..crop.crop_objects import _input_fingerprint


def reference_inputs(scan):
    root = config.parsed_images_dir(scan)
    return {'geometry': json.loads(json.dumps(_input_fingerprint(scan, False))),
            'evidence': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(root.rglob('*'))
                         if p.is_file() and p.suffix in {'.jpg', '.json'}}}
