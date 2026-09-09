# Preprocessing

This package keeps the ported preprocessing code separate from the current
pipeline.

The project owns `scene_data.py` and `object_images.py`. These modules provide
the small interface used by the extract and crop stages. Project settings, such
as `LR_ENLARGED_CROP_OBJECTS`, belong in these adapters.

The `vendor/litereality` package contains code ported from earlier LiteReality
implementations. Its module names state what each file does:

| Module | Role |
| --- | --- |
| `scene_preprocessing.py` | Parses scenes, crops object images, and prepares camera data. |
| `object_image_extraction.py` | Projects point clouds, ranks views, and writes object crops. |
| `roomplan.py` | Reads RoomPlan USD files and extracts RGBD data. |
| `scanner_capture.py` | Loads ScannerApp frames and camera data. |

Keep pipeline control flow and project settings outside `vendor`. Keep fixes to
the ported algorithms small so they remain easy to compare with the earlier
code.
