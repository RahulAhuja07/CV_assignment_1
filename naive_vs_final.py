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
    templates = [str(template_path) for template_path in TEMPLATES if template_path.exists()]
    pipeline = ObjectRecognitionPipeline(templates, str(query_path))

    # Show the raw matches before the later checks remove bad matches.
    front_template = pipeline.templates[0]
    pipeline.template = front_template['img']
    pipeline.template_gray = front_template['gray']
    pipeline.template_keypoints = front_template['keypoints']
    pipeline.template_descriptors = front_template['descriptors']
    raw_matches = pipeline.find_feature_matches()

    display_matches = [cv2.DMatch(_queryIdx=template_index, _trainIdx=query_index, _distance=0)
                       for template_index, query_index in raw_matches]
    raw_match_image = cv2.drawMatches(
        front_template['img'], front_template['keypoints'],
        pipeline.query, pipeline.query_keypoints,
        display_matches, None,
        matchColor=(0, 0, 255), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Run all checks and draw the final detections.
    pipeline.detect_objects()
    final_detection_image = pipeline.query.copy()
    for detection_number, detection in enumerate(pipeline.detections, 1):
        bounding_box = detection['box'].astype(int)
        box_color = (0, 255, 255) if detection.get('source') == 'yellow_red_fallback' else (0, 255, 0)
        cv2.polylines(final_detection_image, [bounding_box], True, box_color, 3)
        label_position = tuple(np.min(bounding_box, axis=0))
        cv2.putText(final_detection_image, f'Maggi {detection_number}', label_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, box_color, 2, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(cv2.cvtColor(raw_match_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Naive matching: {len(raw_matches)} raw correspondences\n'
                       f"(after Lowe's ratio test, before Hough/RANSAC)",
                       fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(final_detection_image, cv2.COLOR_BGR2RGB))
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
