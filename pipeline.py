"""
Multi-Instance Object Recognition in Cluttered Scenes
Assignment 1 - Computer Vision

Complete pipeline for detecting multiple instances of a template object in a
cluttered scene, plus a batch entry point (main()) that runs it over every
image in query_images/.

Pipeline stages (see ObjectRecognitionPipeline.detect_objects, which runs
these in order for every template):
  1. SIFT feature extraction (OpenCV) on the template(s) and query image
     -- done once up front in __init__.
  2. manual_feature_matching()      -- from-scratch nearest-neighbour search
                                        + Lowe's ratio test.
  3. generalized_hough_transform()  -- each match votes for an object pose
                                        (x, y, scale, angle) in a 4D
                                        accumulator; consistent votes cluster
                                        into candidate peaks.
  4. verify_detections()            -- RANSAC affine fit + least-squares
                                        refit per candidate peak, plus
                                        geometric sanity checks (shear,
                                        area, bounds).
  5. post_filter_detections()       -- drops weak/scale-outlier detections;
     color_fallback_detections()       adds colour-based candidates SIFT
                                        missed entirely.
  6. nms_suppress()                 -- IoU-based Non-Maximum Suppression so
                                        one box remains per physical object.

Run with: python pipeline.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class ObjectRecognitionPipeline:
    """Multi-instance object recognition: match, cluster, verify, deduplicate.

    See the module docstring for the six-stage pipeline this class
    implements. All of the matching/clustering/verification math is
    implemented from scratch on top of OpenCV's SIFT keypoints/descriptors.
    """

    def __init__(self, template_path: str | List[str], query_path: str):
        """
        Load the query image and one or more templates, and extract SIFT
        features for all of them up front (this is the only step that uses
        an OpenCV algorithm beyond basic image I/O).

        Args:
            template_path: Path to a single template image, or a list of
                paths (e.g. front + back views of the same object) to be
                tried independently by detect_objects().
            query_path: Path to the cluttered scene to search.
        """
        # Support single template path or list of template paths (front/back)
        if isinstance(template_path, str):
            template_paths = [template_path]
        else:
            template_paths = list(template_path)

        self.query = cv2.imread(query_path)
        if self.query is None:
            raise ValueError("Could not load query image")

        # Convert query to grayscale and extract its features once
        self.query_gray = cv2.cvtColor(self.query, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.SIFT_create()
        # Reproducible RANSAC makes a detection count stable between runs.
        self.rng = np.random.default_rng(42)
        print("Extracting SIFT features for query...")
        self.query_kp, self.query_desc = self.sift.detectAndCompute(self.query_gray, None)
        if self.query_desc is None:
            raise ValueError("Could not extract SIFT features from query")

        # Load and extract features for each template
        self.templates = []
        print("Extracting SIFT features for templates...")
        for tp in template_paths:
            img = cv2.imread(tp)
            if img is None:
                raise ValueError(f"Could not load template: {tp}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = self.sift.detectAndCompute(gray, None)
            if desc is None:
                raise ValueError(f"Could not extract SIFT features from template: {tp}")
            h, w = img.shape[:2]
            self.templates.append({
                'path': tp,
                'img': img,
                'gray': gray,
                'kp': kp,
                'desc': desc,
                'h': h,
                'w': w,
                'center': np.array([w/2.0, h/2.0])
            })

        print(f"Query features: {len(self.query_kp)} keypoints")
        for i, t in enumerate(self.templates, 1):
            print(f"Template {i}: {t['path']} -> {len(t['kp'])} keypoints")
        
        self.detections = []
    
    def manual_feature_matching(self, max_features: int = 10000, ratio_threshold: float = 0.9) -> List[Tuple[int, int]]:
        """
        Manually implemented feature matching using Lowe's ratio test
        
        Args:
            max_features: Maximum features to use for speed
            ratio_threshold: Lowe's ratio threshold
            
        Returns:
            List of (template_idx, query_idx) matches
        """
        print("\n[1] Matching features (manual implementation)...")
        
        template_desc = self.template_desc.astype(np.float32)
        query_desc = self.query_desc.astype(np.float32)
        
        # Subsample for speed ONLY if really large
        if len(template_desc) > max_features:
            idx = np.random.choice(len(template_desc), max_features, replace=False)
            template_desc = template_desc[idx]
            t_indices = np.sort(idx)
        else:
            t_indices = np.arange(len(template_desc))
        
        if len(query_desc) > max_features:
            idx = np.random.choice(len(query_desc), max_features, replace=False)
            query_desc = query_desc[idx]
            q_indices = np.sort(idx)
        else:
            q_indices = np.arange(len(query_desc))
        
        matches = []
        
        # Match each template descriptor
        for local_t_idx, t_desc in enumerate(template_desc):
            # Compute distances to all query descriptors
            distances = np.linalg.norm(query_desc - t_desc, axis=1)
            
            # Find two nearest neighbors
            nearest_idx = np.argsort(distances)[:2]
            dist1 = distances[nearest_idx[0]]
            dist2 = distances[nearest_idx[1]]
            
            # Lowe's ratio test
            if dist1 < ratio_threshold * dist2:
                global_t_idx = t_indices[local_t_idx]
                global_q_idx = q_indices[nearest_idx[0]]
                matches.append((global_t_idx, global_q_idx))
        
        print(f"Found {len(matches)} matches after Lowe's ratio test")
        return matches
    
    def generalized_hough_transform(self, matches: List[Tuple[int, int]]) -> List[Dict]:
        """
        Implement Generalized Hough Transform for 4D space (x, y, scale, angle)
        
        Args:
            matches: List of (template_idx, query_idx) matches
            
        Returns:
            List of detected object instances with their parameters
        """
        print("\n[2] Applying Generalized Hough Transform...")
        
        template_h, template_w = self.template.shape[:2]
        template_center = np.array([template_w / 2.0, template_h / 2.0])
        
        # 4D Hough accumulator: (x_bin, y_bin, scale_bin, angle_bin) -> list of votes.
        # Bin sizes are a tolerance trade-off: too fine and correct votes for
        # the same object scatter across neighbouring bins and never form a
        # peak; too coarse and unrelated matches start colliding into the
        # same bin. These values were tuned empirically for this object size.
        accumulator = {}
        bin_size = 100   # pixels per spatial bin (x, y)
        scale_bins = 6   # quantizes the 0.1x-5.0x scale range below
        angle_bins = 12  # 30 degrees per bin
        
        print(f"Processing {len(matches)} matches...")
        
        for t_idx, q_idx in matches:
            t_kp = self.template_kp[t_idx]
            q_kp = self.query_kp[q_idx]
            
            # Feature positions
            t_pos = np.array(t_kp.pt)
            q_pos = np.array(q_kp.pt)
            
            # Estimate scale and rotation
            # Scale is relative to template size
            q_size = q_kp.size
            t_size = t_kp.size + 1e-6
            scale = q_size / t_size
            
            # Clamp scale to reasonable range
            scale = np.clip(scale, 0.1, 5.0)
            
            angle_diff = (q_kp.angle - t_kp.angle) % 360
            
            # Quantize scale and angle
            # Map 0.1-5.0 range to scale_bins
            scale_normalized = (scale - 0.1) / (5.0 - 0.1)  # 0-1
            scale_bin = int(np.clip(scale_normalized * scale_bins, 0, scale_bins - 1))
            angle_bin = int(angle_diff / 30.0) % angle_bins  # 30 degree bins
            
            # Predict object center
            offset = t_pos - template_center
            angle_rad = np.radians(angle_diff)
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                         [np.sin(angle_rad), np.cos(angle_rad)]])
            rotated_offset = scale * (R @ offset)
            predicted_center = q_pos - rotated_offset
            
            # Quantize location
            x_bin = int(predicted_center[0] / bin_size)
            y_bin = int(predicted_center[1] / bin_size)
            
            # Vote
            bin_key = (x_bin, y_bin, scale_bin, angle_bin)
            if bin_key not in accumulator:
                accumulator[bin_key] = []
            
            accumulator[bin_key].append({
                'center': predicted_center,
                'scale': scale,
                'angle': angle_diff,
                'matches': [(t_idx, q_idx)]
            })
        
        # Find peaks
        peaks = []
        for bin_key, votes in accumulator.items():
            if len(votes) >= 1:  # Minimum 1 vote for now
                centers = np.array([v['center'] for v in votes])
                scales = np.array([v['scale'] for v in votes])
                angles = np.array([v['angle'] for v in votes])
                
                # Collect ALL matches from all votes in this bin
                all_matches = []
                for v in votes:
                    all_matches.extend(v['matches'])
                
                peak = {
                    'center': np.median(centers, axis=0),
                    'scale': np.median(scales),
                    'angle': np.median(angles),
                    'votes': len(votes),
                    'matches': all_matches  # All matches that voted for this bin
                }
                
                peaks.append(peak)
        
        peaks = sorted(peaks, key=lambda p: p['votes'], reverse=True)
        print(f"Found {len(peaks)} candidate peaks in Hough space")
        
        return peaks
    
    def ransac_affine_fit(self, src_pts: np.ndarray, dst_pts: np.ndarray,
                         iterations: int = 1000, threshold: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robust Affine Transformation Estimation using RANSAC (from scratch)
        
        Args:
            src_pts: Source points (template)
            dst_pts: Destination points (query)
            iterations: Number of RANSAC iterations
            threshold: Inlier distance threshold
            
        Returns:
            affine_matrix: 2x3 affine transformation
            inlier_mask: Boolean mask of inliers
        """
        N = len(src_pts)
        best_matrix = None
        best_inliers = None
        best_count = 0
        
        print(f"        RANSAC: {N} points, {iterations} iterations")
        
        for iteration in range(iterations):
            # A 2D affine transform has 6 unknown coefficients, and each
            # point correspondence contributes 2 equations (one for x, one
            # for y) -- so the minimal sample that exactly determines the
            # system is 3 points.
            sample_idx = self.rng.choice(N, 3, replace=False)

            try:
                # Build system Ax = b for 3 point correspondences
                src_sample = src_pts[sample_idx]
                dst_sample = dst_pts[sample_idx]
                
                # Solve using least squares
                A = np.zeros((6, 6))
                b = np.zeros(6)
                
                for i in range(3):
                    x, y = src_sample[i]
                    u, v = dst_sample[i]
                    A[2*i, :] = [x, y, 1, 0, 0, 0]
                    A[2*i+1, :] = [0, 0, 0, x, y, 1]
                    b[2*i] = u
                    b[2*i+1] = v
                
                coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
                matrix = np.array([
                    [coeffs[0], coeffs[1], coeffs[2]],
                    [coeffs[3], coeffs[4], coeffs[5]]
                ])
                
                # A packet is rigid and planar.  Reject reflections and transforms
                # with implausibly large area changes before scoring them.
                det = np.linalg.det(matrix[:2, :2])
                if det <= 0.002 or det > 20:
                    continue
                
                # Count inliers
                src_h = np.hstack([src_pts, np.ones((N, 1))])
                dst_pred = src_h @ matrix.T
                errors = np.linalg.norm(dst_pts - dst_pred, axis=1)
                inliers = errors < threshold
                inlier_count = np.sum(inliers)
                
                if inlier_count > best_count:
                    best_count = inlier_count
                    best_matrix = matrix
                    best_inliers = inliers
                    print(f"        Iteration {iteration}: {inlier_count} inliers, det={det:.3f}")
            
            except Exception as e:
                continue
        
        # Refit the winning model to every inlier.  This makes the projected box
        # substantially less jittery than retaining a random three-point sample.
        if best_matrix is not None and best_count >= 3:
            in_src = src_pts[best_inliers]
            in_dst = dst_pts[best_inliers]
            design = []
            target = []
            for (x, y), (u, v) in zip(in_src, in_dst):
                design.extend([[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]])
                target.extend([u, v])
            coeffs = np.linalg.lstsq(np.asarray(design), np.asarray(target), rcond=None)[0]
            best_matrix = np.array([[coeffs[0], coeffs[1], coeffs[2]],
                                    [coeffs[3], coeffs[4], coeffs[5]]])
            src_h = np.hstack([src_pts, np.ones((N, 1))])
            errors = np.linalg.norm(dst_pts - src_h @ best_matrix.T, axis=1)
            best_inliers = errors < threshold
            best_count = int(np.sum(best_inliers))

        print(f"        Best: {best_count} inliers")
        return best_matrix, best_inliers
    
    def verify_detections(self, peaks: List[Dict]) -> List[Dict]:
        """
        Verify detections using RANSAC affine fitting
        
        Args:
            peaks: Candidate peaks from Hough transform
            
        Returns:
            List of verified detections
        """
        print("\n[3] Verifying detections with RANSAC...")
        
        detections = []
        template_h, template_w = self.template.shape[:2]

        # Peaks are already sorted by vote count; only the strongest few can
        # plausibly be real objects; the long tail is almost entirely
        # 1-3 vote noise bins that would just waste RANSAC iterations.
        for peak_idx, peak in enumerate(peaks[:10]):
            matches = peak['matches']
            
            # Collect point pairs
            src_pts = []
            dst_pts = []
            
            for t_idx, q_idx in matches:
                if t_idx < len(self.template_kp) and q_idx < len(self.query_kp):
                    src_pts.append(self.template_kp[t_idx].pt)
                    dst_pts.append(self.query_kp[q_idx].pt)
            
            if len(src_pts) < 3:
                print(f"  Peak {peak_idx}: Skipped, only {len(src_pts)} points")
                continue
            
            src_pts = np.array(src_pts)
            dst_pts = np.array(dst_pts)
            
            print(f"  Peak {peak_idx}: Testing with {len(src_pts)} points...")
            
            # Fit affine transformation
            affine_matrix, inlier_mask = self.ransac_affine_fit(src_pts, dst_pts)
            
            if affine_matrix is None:
                print(f"  Peak {peak_idx}: RANSAC returned None")
                continue
            
            if inlier_mask is None:
                print(f"  Peak {peak_idx}: Inlier mask is None")
                continue
            
            inlier_count = np.sum(inlier_mask)
            print(f"  Peak {peak_idx}: {inlier_count} inliers")

            # 3 inliers is the RANSAC minimum sample size itself, so it proves
            # nothing beyond "a transform exists"; requiring 4+ here means at
            # least one point independently agreed with the fitted model.
            if inlier_count < 4:
                continue

            # Prevent a small accidental correspondence set from producing an
            # extremely sheared or elongated "packet".  Perspective can change
            # the aspect ratio a little, but not by several times on this setup.
            linear = affine_matrix[:, :2]
            singular_values = np.linalg.svd(linear, compute_uv=False)
            if singular_values[1] <= 1e-6 or singular_values[0] / singular_values[1] > 1.75:
                print(f"  Peak {peak_idx}: rejected (excessive affine shear)")
                continue

            # Project template corners
            corners = np.array([
                [0, 0],
                [template_w, 0],
                [template_w, template_h],
                [0, template_h]
            ], dtype=np.float32)

            corners_h = np.hstack([corners, np.ones((4, 1))])
            projected = corners_h @ affine_matrix.T

            # Shoelace formula for polygon area; a degenerate/near-collinear
            # affine fit can project the template corners onto a sliver of a
            # few pixels, which is never a real packet.
            signed_area = 0.5 * abs(np.sum(projected[:, 0] * np.roll(projected[:, 1], -1)
                                           - projected[:, 1] * np.roll(projected[:, 0], -1)))
            if signed_area < 400:
                print(f"  Peak {peak_idx}: rejected (box too small)")
                continue

            # A packet at the edge of the frame is legitimately half off-screen,
            # so require full bounds only when the box mostly leaves the image;
            # otherwise fall back to an overlap-ratio check below.
            query_h, query_w = self.query.shape[:2]
            inside_x = np.logical_and(projected[:, 0] >= 0, projected[:, 0] <= query_w)
            inside_y = np.logical_and(projected[:, 1] >= 0, projected[:, 1] <= query_h)
            corners_inside = np.sum(np.logical_and(inside_x, inside_y))

            # 2+ corners inside the frame is treated as "clearly present" and
            # accepted outright; otherwise fall through to the overlap check.
            if corners_inside >= 2:
                pass
            else:
                # Compute axis-aligned bbox of projected quad
                box_min = np.min(projected, axis=0)
                box_max = np.max(projected, axis=0)

                # Intersection with image
                inter_min = np.maximum(box_min, [0, 0])
                inter_max = np.minimum(box_max, [query_w, query_h])
                inter_wh = inter_max - inter_min
                inter_wh = np.maximum(inter_wh, 0)
                intersection = inter_wh[0] * inter_wh[1]

                box_area = max(1.0, (box_max[0] - box_min[0]) * (box_max[1] - box_min[1]))
                overlap_ratio = intersection / box_area

                # Accept if substantial overlap with image (allowing edge/partial views)
                # or accept partial views with slightly fewer inliers
                if overlap_ratio < 0.12:
                    continue
                if inlier_count < 3:
                    continue
            
            detection = {
                'box': projected,
                'affine': affine_matrix,
                'inliers': inlier_count,
                'scale': peak['scale'],
                'angle': peak['angle']
            }
            
            detections.append(detection)
            print(f"  Detection {len(detections)}: {inlier_count} inliers, scale={peak['scale']:.2f}x")
        
        return detections
    
    def nms_suppress(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Non-Maximum Suppression to remove duplicates
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold
            
        Returns:
            Filtered detections
        """
        print(f"\n[4] Applying Non-Maximum Suppression...")
        print(f"Detections before NMS: {len(detections)}")
        
        if len(detections) == 0:
            return detections
        
        detections = sorted(detections, key=lambda d: d['inliers'], reverse=True)
        kept = []
        
        while len(detections) > 0:
            current = detections[0]
            kept.append(current)
            detections = detections[1:]
            
            remaining = []
            for det in detections:
                # Simple IoU based on axis-aligned bounding boxes
                box1_min = np.min(current['box'], axis=0)
                box1_max = np.max(current['box'], axis=0)
                box2_min = np.min(det['box'], axis=0)
                box2_max = np.max(det['box'], axis=0)
                
                inter_min = np.maximum(box1_min, box2_min)
                inter_max = np.minimum(box1_max, box2_max)
                inter = np.maximum(0, inter_max - inter_min)
                intersection = np.prod(inter)
                
                area1 = np.prod(box1_max - box1_min)
                area2 = np.prod(box2_max - box2_min)
                union = area1 + area2 - intersection
                
                iou = intersection / (union + 1e-6)
                
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        print(f"Detections after NMS: {len(kept)}")
        return kept
    
    def detect_objects(self):
        """Run complete detection pipeline"""
        print("\n" + "="*60)
        print("MULTI-INSTANCE OBJECT RECOGNITION PIPELINE")
        print("="*60)

        all_detections = []

        # Run detection for each template (front/back or augmentations)
        for t_idx, tdata in enumerate(self.templates):
            print(f"\nProcessing template {t_idx+1}: {tdata['path']}")
            # Set active template attributes expected by existing methods
            self.template = tdata['img']
            self.template_gray = tdata['gray']
            self.template_kp = tdata['kp']
            self.template_desc = tdata['desc']
            # Run matching -> Hough -> verify
            matches = self.manual_feature_matching()
            if len(matches) < 3:
                print("Not enough matches for this template")
                continue

            peaks = self.generalized_hough_transform(matches)
            if len(peaks) == 0:
                print("No peaks found for this template")
                continue

            dets = self.verify_detections(peaks)
            # tag detections with template id
            for d in dets:
                d['template_idx'] = t_idx
                d['template_path'] = tdata['path']
            all_detections.extend(dets)

        if len(all_detections) == 0:
            print("No detections across templates")
            return

        # Smart post-filter across all templates
        filtered = self.post_filter_detections(all_detections)

        # A flat, glossy packet can have too few stable SIFT points (especially
        # near an image edge).  Add only unmatched regions having Maggi's strong
        # yellow-and-red packaging signature; this is deliberately a fallback,
        # not a replacement for geometric feature verification.
        filtered.extend(self.color_fallback_detections(filtered))

        # Final NMS merge
        self.detections = self.nms_suppress(filtered)

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Total objects detected: {len(self.detections)}")
        for i, det in enumerate(self.detections, 1):
            print(f"  Object {i}: {det['inliers']} inliers, scale={det['scale']:.2f}x, angle={det['angle']:.1f}°")
        print("="*60)

    def color_fallback_detections(self, detections: List[Dict]) -> List[Dict]:
        """Find unmatched Maggi-like yellow/red packet regions.

        The size, aspect-ratio and red-content checks prevent yellow notebooks
        and other desk items from being counted as packets.
        """
        hsv = cv2.cvtColor(self.query, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (18, 100, 80), (42, 255, 255))
        red = cv2.bitwise_or(cv2.inRange(hsv, (0, 100, 60), (10, 255, 255)),
                             cv2.inRange(hsv, (170, 100, 60), (180, 255, 255)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        added = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 7000 <= area <= 50000:
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (width, height), _ = rect
            short_side = max(1.0, min(width, height))
            if max(width, height) / short_side > 2.2:
                continue

            box = cv2.boxPoints(rect)
            region = np.zeros(yellow.shape, dtype=np.uint8)
            cv2.fillConvexPoly(region, box.astype(np.int32), 255)
            red_fraction = cv2.countNonZero(cv2.bitwise_and(red, region)) / max(1, cv2.countNonZero(region))
            if red_fraction < 0.08:
                continue

            # Avoid adding a colour candidate already localized by SIFT.
            candidate_min, candidate_max = np.min(box, axis=0), np.max(box, axis=0)
            duplicate = False
            for det in detections:
                det_min, det_max = np.min(det['box'], axis=0), np.max(det['box'], axis=0)
                inter = np.maximum(0, np.minimum(candidate_max, det_max) - np.maximum(candidate_min, det_min))
                intersection = float(np.prod(inter))
                union = float(np.prod(candidate_max - candidate_min) + np.prod(det_max - det_min) - intersection)
                if intersection / max(union, 1.0) >= 0.20:
                    duplicate = True
                    break
            if not duplicate:
                added.append({'box': box, 'inliers': 0, 'scale': 0.0, 'angle': 0.0,
                              'source': 'yellow_red_fallback'})
        if added:
            print(f"Colour fallback: added {len(added)} unmatched packet candidate(s)")
        return added

    def post_filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Apply smart filtering to remove weak false positives while keeping partial/edge detections.

        Rules:
        - Keep detections with inliers >= 5
        - If inliers == 4, require overlap_ratio >= 0.18
        - If inliers == 3, require overlap_ratio >= 0.35
        - Remove detections with scale very different from median scale (factor > 2.5)
        """
        if not detections:
            return detections

        scales = np.array([d['scale'] for d in detections])
        median_scale = np.median(scales)

        kept = []
        for d in detections:
            if d.get('source') == 'yellow_red_fallback':
                kept.append(d)
                continue
            inliers = d['inliers']
            box = d['box']
            box_min = np.min(box, axis=0)
            box_max = np.max(box, axis=0)
            query_h, query_w = self.query.shape[:2]
            inter_min = np.maximum(box_min, [0, 0])
            inter_max = np.minimum(box_max, [query_w, query_h])
            inter_wh = np.maximum(inter_max - inter_min, 0)
            intersection = inter_wh[0] * inter_wh[1]
            box_area = max(1.0, (box_max[0] - box_min[0]) * (box_max[1] - box_min[1]))
            overlap_ratio = intersection / box_area

            # Scale sanity
            if d['scale'] > median_scale * 2.5 or d['scale'] < median_scale / 2.5:
                # reject extreme scale outliers unless very high inliers
                if inliers < 10:
                    continue

            if inliers >= 5:
                kept.append(d)
                continue

            if inliers == 4 and overlap_ratio >= 0.18:
                kept.append(d)
                continue

            if inliers == 3 and overlap_ratio >= 0.35:
                kept.append(d)
                continue

            # else discard
        print(f"Post-filter: {len(kept)} kept of {len(detections)}")
        return kept
    
    def visualize_results(self, output_path: str = None):
        """Save an annotated detection image without depending on a GUI backend."""
        result = self.query.copy()
        
        for index, detection in enumerate(self.detections, 1):
            box = detection['box'].astype(int)
            colour = (0, 255, 255) if detection.get('source') == 'yellow_red_fallback' else (0, 255, 0)
            cv2.polylines(result, [box], True, colour, 3)
            anchor = tuple(np.min(box, axis=0))
            cv2.putText(result, f"Maggi {index}", anchor, cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, colour, 2, cv2.LINE_AA)
        
        if output_path:
            if not cv2.imwrite(output_path, result):
                raise IOError(f"Could not write visualization: {output_path}")
            print(f"\nVisualization saved to: {output_path}")


def main():
    """Batch entry point: run the pipeline over every image in query_images/.

    Uses both the front and back template views (a back-facing packet has
    very different SIFT features from the front view) and writes one
    annotated PNG per query image plus a combined JSON summary.
    """
    workspace = Path(__file__).resolve().parent

    cropped = [workspace / 'Maggi_cropped.jpeg', workspace / 'Maggi_back_cropped.jpeg']
    originals = [workspace / 'Maggi.jpeg', workspace / 'Maggi_back.jpeg']
    templates = [p for p in cropped if p.exists()] or [p for p in originals if p.exists()]

    query_dir = workspace / 'query_images'
    outdir = workspace / 'batch_results'
    vizdir = outdir / 'visualizations'
    vizdir.mkdir(parents=True, exist_ok=True)

    query_files = []
    for pattern in ('*.jpg', '*.jpeg', '*.png'):
        query_files.extend(sorted(query_dir.glob(pattern)))

    if not query_files:
        print(f'No query images found in {query_dir}')
        return

    summary = []
    for qpath in query_files:
        print(f'Processing {qpath.name} ...')
        try:
            pipeline = ObjectRecognitionPipeline([str(p) for p in templates], str(qpath))
            pipeline.detect_objects()
            out_png = vizdir / f'results_{qpath.stem}.png'
            pipeline.visualize_results(str(out_png))

            dets = []
            for det in pipeline.detections:
                dets.append({
                    'inliers': int(det.get('inliers', 0)),
                    'scale': float(det.get('scale', 0.0)),
                    'angle': float(det.get('angle', 0.0)),
                    'box': [[float(x), float(y)] for x, y in det['box'].tolist()]
                })

            summary.append({
                'image': qpath.name,
                'num_detections': len(pipeline.detections),
                'detections': dets,
                'output_image': str(out_png)
            })

        except Exception as e:
            print(f'Error processing {qpath.name}:', e)
            summary.append({'image': qpath.name, 'error': str(e)})

    summary_path = outdir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nBatch run complete. Summary saved to', summary_path)
    print('Per-image visualizations saved in', vizdir)


if __name__ == "__main__":
    main()
