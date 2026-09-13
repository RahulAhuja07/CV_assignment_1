"""
Naive Feature Matching vs. Final Processed Detections
Assignment 1 - Computer Vision

Builds the comparison the report asks for: a "naive" match panel showing every
raw feature correspondence (post Lowe's-ratio-test, pre Hough/RANSAC) as a
tangle of lines across the cluttered scene, next to the pipeline's final,
geometrically-verified bounding boxes for the same image. The gap between the
two panels is the point -- it shows what the Generalized Hough Transform and
RANSAC step actually remove.

Run with: python naive_vs_final.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline import ObjectRecognitionPipeline

WORKSPACE = Path(__file__).resolve().parent
TEMPLATES = [WORKSPACE / 'Maggi_cropped.jpeg', WORKSPACE / 'Maggi_back_cropped.jpeg']
OUTDIR = WORKSPACE / 'naive_vs_final'
OUTDIR.mkdir(exist_ok=True)


def build_comparison(query_path: Path, out_path: Path):
    print(f'\n=== Naive vs. final comparison for {query_path.name} ===')
    templates = [str(p) for p in TEMPLATES if p.exists()]
    pipeline = ObjectRecognitionPipeline(templates, str(query_path))

    # Naive panel: raw Lowe's-ratio matches for the front template, before any
    # geometric verification (Hough voting or RANSAC) has had a chance to
    # reject the ones that are just coincidental background noise.
    front = pipeline.templates[0]
    pipeline.template = front['img']
    pipeline.template_gray = front['gray']
    pipeline.template_kp = front['kp']
    pipeline.template_desc = front['desc']
    raw_matches = pipeline.manual_feature_matching()

    dmatches = [cv2.DMatch(_queryIdx=t_idx, _trainIdx=q_idx, _distance=0)
                for t_idx, q_idx in raw_matches]
    naive_img = cv2.drawMatches(
        front['img'], front['kp'],
        pipeline.query, pipeline.query_kp,
        dmatches, None,
        matchColor=(0, 0, 255), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Final panel: the full pipeline (both templates, Hough + RANSAC + NMS).
    pipeline.detect_objects()
    final_img = pipeline.query.copy()
    for idx, det in enumerate(pipeline.detections, 1):
        box = det['box'].astype(int)
        colour = (0, 255, 255) if det.get('source') == 'yellow_red_fallback' else (0, 255, 0)
        cv2.polylines(final_img, [box], True, colour, 3)
        anchor = tuple(np.min(box, axis=0))
        cv2.putText(final_img, f'Maggi {idx}', anchor, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, colour, 2, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(cv2.cvtColor(naive_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Naive matching: {len(raw_matches)} raw correspondences\n'
                       f"(after Lowe's ratio test, before Hough/RANSAC)",
                       fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Final result: {len(pipeline.detections)} verified detections\n'
                       f'(Generalized Hough Transform + RANSAC + NMS)',
                       fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    build_comparison(WORKSPACE / 'query_images' / 'Cluster.jpeg', OUTDIR / 'naive_vs_final_cluster.png')
    build_comparison(WORKSPACE / 'query_images' / 'img-3.jpeg', OUTDIR / 'naive_vs_final_img-3.png')
