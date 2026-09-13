#!/usr/bin/env python3
"""
Verification script to check detected packets against manual count
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

# Load images
template_path = '/Users/rahulahuja07/Desktop/CV_assignment1/Maggi.jpeg'
query_path = '/Users/rahulahuja07/Desktop/CV_assignment1/query_images/Cluster.jpeg'
template = cv2.imread(template_path)
query = cv2.imread(query_path)

# Run pipeline
from pipeline import ObjectRecognitionPipeline

pipeline = ObjectRecognitionPipeline(template_path, query_path)
pipeline.detect_objects()

# Create detailed visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Left: Original query image
axes[0].imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Query Image', fontsize=16, fontweight='bold')
axes[0].axis('on')
axes[0].grid(True, alpha=0.3)

# Right: Query with detected bounding boxes
result = query.copy()
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]

for idx, detection in enumerate(pipeline.detections):
    box = detection['box'].astype(int)
    color = colors[idx % len(colors)]
    cv2.polylines(result, [box], True, color, 3)
    
    # Add label with detection info
    center = box.mean(axis=0).astype(int)
    inliers = detection['inliers']
    scale = detection['scale']
    angle = detection['angle']
    cv2.putText(result, f"Obj{idx+1}: {inliers}in", 
                (center[0]-30, center[1]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Detection Results - {len(pipeline.detections)} Packets Found', 
                   fontsize=16, fontweight='bold')
axes[1].axis('on')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/rahulahuja07/Desktop/CV_assignment1/verification_comparison.png', 
            dpi=150, bbox_inches='tight')
print("Saved verification_comparison.png")

# Print detailed statistics
print("\n" + "="*70)
print("DETECTION VERIFICATION REPORT")
print("="*70)
print(f"\nTotal Maggi packets detected: {len(pipeline.detections)}")
print(f"Total feature matches found: {len(pipeline.detections) if hasattr(pipeline, '_matches') else 'N/A'}")

for i, det in enumerate(pipeline.detections, 1):
    print(f"\nPacket {i}:")
    print(f"  - Inliers (point correspondences): {det['inliers']}")
    print(f"  - Scale factor: {det['scale']:.3f}x")
    print(f"  - Rotation angle: {det['angle']:.1f}°")
    print(f"  - Bounding box corners:")
    for j, corner in enumerate(det['box']):
        print(f"    Corner {j+1}: ({corner[0]:.1f}, {corner[1]:.1f})")

print("\n" + "="*70)
print("MANUAL VERIFICATION STEPS:")
print("="*70)
print("1. Open 'Cluster.jpeg' and manually count all visible Maggi packets")
print("2. Compare count with detected packets above")
print("3. Check 'verification_comparison.png' for visual inspection")
print("4. Verify bounding boxes align with packet locations")
print("5. Check if any packets are missed (false negatives)")
print("6. Check if any non-packets are marked (false positives)")
print("="*70)

# Display
plt.show()
