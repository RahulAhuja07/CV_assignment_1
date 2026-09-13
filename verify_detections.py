#!/usr/bin/env python3
"""Draw detected Maggi packets and print a simple verification report."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from pipeline import ObjectRecognitionPipeline


def create_verification_image(scene_image, detections, output_path: Path):
    """Save the original scene beside the scene with detection boxes."""
    figure, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Show the input image first.
    axes[0].imshow(cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Scene Image', fontsize=16, fontweight='bold')
    axes[0].axis('on')
    axes[0].grid(True, alpha=0.3)

    marked_image = scene_image.copy()
    box_colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for detection_index, detection in enumerate(detections):
        bounding_box = detection['box'].astype(int)
        box_color = box_colors[detection_index % len(box_colors)]
        cv2.polylines(marked_image, [bounding_box], True, box_color, 3)
        label_center = bounding_box.mean(axis=0).astype(int)
        cv2.putText(marked_image, f"Obj{detection_index + 1}: {detection['inliers']}in",
                    (label_center[0] - 30, label_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    axes[1].imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Detection Results - {len(detections)} Packets Found',
                      fontsize=16, fontweight='bold')
    axes[1].axis('on')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    print(f"Saved {output_path.name}")


def print_verification_report(detections):
    """Print the detection details and manual checking steps."""
    print("\n" + "=" * 70)
    print("DETECTION VERIFICATION REPORT")
    print("=" * 70)
    print(f"\nTotal Maggi packets detected: {len(detections)}")
    for detection_number, detection in enumerate(detections, 1):
        print(f"\nPacket {detection_number}:")
        print(f"  - Matching points: {detection['inliers']}")
        print(f"  - Scale factor: {detection['scale']:.3f}x")
        print(f"  - Rotation angle: {detection['angle']:.1f}°")
        print("  - Bounding-box corners:")
        for corner_number, corner in enumerate(detection['box'], 1):
            print(f"    Corner {corner_number}: ({corner[0]:.1f}, {corner[1]:.1f})")

    print("\n" + "=" * 70)
    print("MANUAL CHECKING STEPS:")
    print("=" * 70)
    print("1. Open Cluster.jpeg and count the visible Maggi packets.")
    print("2. Compare your count with the number above.")
    print("3. Check verification_comparison.png.")
    print("4. Make sure each box is around a packet.")
    print("5. Check for missed packets and incorrect boxes.")
    print("=" * 70)


def main():
    """Run the detector on Cluster.jpeg and save a comparison image."""
    workspace = Path(__file__).resolve().parent
    template_path = workspace / 'Maggi.jpeg'
    scene_path = workspace / 'query_images' / 'Cluster.jpeg'
    output_path = workspace / 'verification_comparison.png'
    scene_image = cv2.imread(str(scene_path))
    if scene_image is None:
        raise ValueError(f"Could not load scene image: {scene_path}")

    recognition_pipeline = ObjectRecognitionPipeline(str(template_path), str(scene_path))
    recognition_pipeline.detect_objects()
    create_verification_image(scene_image, recognition_pipeline.detections, output_path)
    print_verification_report(recognition_pipeline.detections)


if __name__ == '__main__':
    main()
