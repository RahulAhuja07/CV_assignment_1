# Multi-Instance Object Recognition in Cluttered Scenes

Detects and localizes every instance of a Maggi noodles packet in cluttered
desk photos, using SIFT for feature extraction (OpenCV) and a from-scratch
implementation of feature matching, a 4D Generalized Hough Transform, RANSAC
affine fitting, and IoU-based Non-Maximum Suppression.

## Project layout

```
pipeline.py               Core pipeline + batch entry point (run this)
naive_vs_final.py          Generates the naive-match vs. final-detection comparison figures
verify_detections.py       Diagnostic script: original image vs. detections, side by side

Maggi.jpeg                 Original front-view template photo (isolated packet)
Maggi_back.jpeg            Original back-view template photo
Maggi_cropped.jpeg         Front template cropped to the packet (used by the pipeline)
Maggi_back_cropped.jpeg    Back template cropped to the packet (used by the pipeline)

query_images/               All cluttered query photos (pipeline input)
  Cluster.jpeg
  img-1.jpeg ... img-6.jpeg

batch_results/               Output of pipeline.py
  summary.json                Per-image detection data (box corners, inliers, scale, angle)
  visualizations/              One annotated PNG per query image
  results.png                  Standalone copy of an earlier single-image run

naive_vs_final/               Output of naive_vs_final.py
  naive_vs_final_cluster.png
  naive_vs_final_img-3.png

report/                       LaTeX report (images added separately, see report/images/README.md)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy matplotlib
```

## Running

```bash
python pipeline.py          # batch-processes every image in query_images/, writes batch_results/
python naive_vs_final.py    # builds the naive-vs-final comparison figures used in the report
python verify_detections.py # quick visual sanity check against Cluster.jpeg
```

## How it works

1. **SIFT feature extraction** (OpenCV) on the template(s) and the query image.
2. **Feature correspondence, from scratch**: for every query descriptor, the two
   nearest template descriptors are found by brute-force Euclidean distance,
   and Lowe's ratio test (`d1 < 0.9 * d2`) rejects ambiguous matches.
3. **4D Generalized Hough Transform**: each match casts one vote for an object
   center (x, y), scale, and orientation, binned into a discrete accumulator.
   Bins that accumulate several consistent votes are candidate detections.
4. **RANSAC + least-squares affine fit**: for each candidate bin, random
   3-point samples solve `Ax = b` for a 2D affine transform; the sample with
   the most inliers under a pixel-distance threshold is refit to all its
   inliers. Determinant checks reject degenerate/reflected transforms.
5. **Greedy extraction**: this runs per template (front and back view) and
   the results are pooled.
6. **Post-filtering + IoU Non-Maximum Suppression** removes weak/duplicate
   detections so one clean box remains per physical packet.

See `report/report.tex` for the full mathematical write-up.
