import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import List, Tuple, Dict


class ObjectRecognitionPipeline:
    """Find several copies of a template in one image."""

    def __init__(self, template_path: str | List[str], query_path: str):
        """
        Load the scene and template images, then extract SIFT features.

        Args:
            template_path: Path to a single template image, or a list of
                paths to try one at a time.
            query_path: Path to the scene image to search.
        """
        # Accept one template path or a list of template paths.
        if isinstance(template_path, str):
            template_paths = [template_path]
        else:
            template_paths = list(template_path)

        self.query = cv2.imread(query_path)
        if self.query is None:
            raise ValueError("Could not load query image")

        # Convert the scene once and find its features.
        self.query_gray = cv2.cvtColor(self.query, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.SIFT_create()
        # A fixed seed gives the same RANSAC result each run.
        self.rng = np.random.default_rng(42)
        print("Extracting SIFT features for query...")
        self.query_keypoints, self.query_descriptors = self.sift.detectAndCompute(self.query_gray, None)
        if self.query_descriptors is None:
            raise ValueError("Could not extract SIFT features from query")

        # Load each template and find its features.
        self.templates = []
        print("Extracting SIFT features for templates...")
        for template_path_item in template_paths:
            template_image = cv2.imread(template_path_item)
            if template_image is None:
                raise ValueError(f"Could not load template: {template_path_item}")
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            template_keypoints, template_descriptors = self.sift.detectAndCompute(template_gray, None)
            if template_descriptors is None:
                raise ValueError(f"Could not extract SIFT features from template: {template_path_item}")
            template_height, template_width = template_image.shape[:2]
            self.templates.append({
                'path': template_path_item,
                'img': template_image,
                'gray': template_gray,
                'keypoints': template_keypoints,
                'descriptors': template_descriptors,
                'height': template_height,
                'width': template_width,
                'center': np.array([template_width / 2.0, template_height / 2.0])
            })

        print(f"Query features: {len(self.query_keypoints)} keypoints")
        for template_number, template_data in enumerate(self.templates, 1):
            print(f"Template {template_number}: {template_data['path']} -> {len(template_data['keypoints'])} keypoints")
        
        self.detections = []
    
    def find_feature_matches(self, max_features: int = 10000, ratio_threshold: float = 0.9) -> List[Tuple[int, int]]:
        """
        Match template features to scene features with Lowe's ratio test.
        
        Args:
            max_features: Maximum number of features to check.
            ratio_threshold: Match-quality limit.
            
        Returns:
            Pairs of template and scene feature indexes.
        """
        print("\n[1] Matching features (manual implementation)...")
        
        template_descriptors = self.template_descriptors.astype(np.float32)
        query_descriptors = self.query_descriptors.astype(np.float32)
        
        # Use a smaller random set only for very large feature lists.
        if len(template_descriptors) > max_features:
            selected_indexes = np.random.choice(len(template_descriptors), max_features, replace=False)
            template_descriptors = template_descriptors[selected_indexes]
            template_indexes = np.sort(selected_indexes)
        else:
            template_indexes = np.arange(len(template_descriptors))
        
        if len(query_descriptors) > max_features:
            selected_indexes = np.random.choice(len(query_descriptors), max_features, replace=False)
            query_descriptors = query_descriptors[selected_indexes]
            query_indexes = np.sort(selected_indexes)
        else:
            query_indexes = np.arange(len(query_descriptors))
        
        matches = []
        
        # Compare every template feature with the scene features.
        for local_template_index, template_descriptor in enumerate(template_descriptors):
            distances = np.linalg.norm(query_descriptors - template_descriptor, axis=1)
            
            # Find the two closest scene features.
            nearest_indexes = np.argsort(distances)[:2]
            best_distance = distances[nearest_indexes[0]]
            second_best_distance = distances[nearest_indexes[1]]
            
            # Keep a match only when the best choice is clearly better.
            if best_distance < ratio_threshold * second_best_distance:
                template_index = template_indexes[local_template_index]
                query_index = query_indexes[nearest_indexes[0]]
                matches.append((template_index, query_index))
        
        print(f"Found {len(matches)} matches after Lowe's ratio test")
        return matches
    
    def find_hough_candidates(self, matches: List[Tuple[int, int]]) -> List[Dict]:
        """
        Group matches that point to the same position, size, and angle.
        
        Args:
            matches: List of (template_idx, query_idx) matches
            
        Returns:
            Possible object locations and their match data.
        """
        print("\n[2] Applying Generalized Hough Transform...")
        
        template_h, template_w = self.template.shape[:2]
        template_center = np.array([template_w / 2.0, template_h / 2.0])
        
        # Each 4D grid cell stores votes for one possible object.
        # These cell sizes allow nearby correct votes to group together.
        accumulator = {}
        bin_size = 100   # pixels per spatial bin (x, y)
        scale_bins = 6   # quantizes the 0.1x-5.0x scale range below
        angle_bins = 12  # 30 degrees per bin
        
        print(f"Processing {len(matches)} matches...")
        
        for template_index, query_index in matches:
            template_keypoint = self.template_keypoints[template_index]
            query_keypoint = self.query_keypoints[query_index]
            
            template_position = np.array(template_keypoint.pt)
            query_position = np.array(query_keypoint.pt)
            
            # Estimate the size and rotation change.
            query_size = query_keypoint.size
            template_size = template_keypoint.size + 1e-6
            scale = query_size / template_size
            
            # Keep only reasonable sizes.
            scale = np.clip(scale, 0.1, 5.0)
            
            angle_difference = (query_keypoint.angle - template_keypoint.angle) % 360
            
            # Put scale and angle into grid cells.
            scale_normalized = (scale - 0.1) / (5.0 - 0.1)  # 0-1
            scale_bin = int(np.clip(scale_normalized * scale_bins, 0, scale_bins - 1))
            angle_bin = int(angle_difference / 30.0) % angle_bins
            
            # Use this match to estimate the object's center.
            offset_from_center = template_position - template_center
            angle_radians = np.radians(angle_difference)
            rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                        [np.sin(angle_radians), np.cos(angle_radians)]])
            rotated_offset = scale * (rotation_matrix @ offset_from_center)
            predicted_center = query_position - rotated_offset
            
            # Put the estimated location into a grid cell.
            x_bin = int(predicted_center[0] / bin_size)
            y_bin = int(predicted_center[1] / bin_size)
            
            # Add this match as a vote for the grid cell.
            bin_key = (x_bin, y_bin, scale_bin, angle_bin)
            if bin_key not in accumulator:
                accumulator[bin_key] = []
            
            accumulator[bin_key].append({
                'center': predicted_center,
                'scale': scale,
                'angle': angle_difference,
                'matches': [(template_index, query_index)]
            })
        
        # Keep grid cells with votes.
        peaks = []
        for bin_key, votes in accumulator.items():
            if len(votes) >= 1:  # Minimum 1 vote for now
                centers = np.array([vote['center'] for vote in votes])
                scales = np.array([vote['scale'] for vote in votes])
                angles = np.array([vote['angle'] for vote in votes])
                
                # Collect every match that voted for this cell.
                all_matches = []
                for vote in votes:
                    all_matches.extend(vote['matches'])
                
                peak = {
                    'center': np.median(centers, axis=0),
                    'scale': np.median(scales),
                    'angle': np.median(angles),
                    'votes': len(votes),
                    'matches': all_matches  # All matches that voted for this bin
                }
                
                peaks.append(peak)
        
        peaks = sorted(peaks, key=lambda candidate: candidate['votes'], reverse=True)
        print(f"Found {len(peaks)} candidate peaks in Hough space")
        
        return peaks
    
    def fit_affine_with_ransac(self, source_points: np.ndarray, destination_points: np.ndarray,
                         iterations: int = 1000, threshold: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find a reliable affine transform with RANSAC.
        
        Args:
            source_points: Points in the template.
            destination_points: Matching points in the scene.
            iterations: Number of random trials.
            threshold: Maximum inlier error in pixels.
            
        Returns:
            affine_matrix: 2x3 affine transformation
            inlier_mask: Boolean mask of inliers
        """
        point_count = len(source_points)
        best_matrix = None
        best_inliers = None
        best_count = 0
        
        print(f"        RANSAC: {point_count} points, {iterations} iterations")
        
        for iteration in range(iterations):
            # Three point pairs are enough to define a 2D affine transform.
            sample_indexes = self.rng.choice(point_count, 3, replace=False)

            try:
                # Build equations from three matching point pairs.
                source_sample = source_points[sample_indexes]
                destination_sample = destination_points[sample_indexes]
                
                coefficient_matrix = np.zeros((6, 6))
                target_values = np.zeros(6)
                
                for i in range(3):
                    x, y = source_sample[i]
                    u, v = destination_sample[i]
                    coefficient_matrix[2*i, :] = [x, y, 1, 0, 0, 0]
                    coefficient_matrix[2*i+1, :] = [0, 0, 0, x, y, 1]
                    target_values[2*i] = u
                    target_values[2*i+1] = v
                
                coefficients = np.linalg.lstsq(coefficient_matrix, target_values, rcond=None)[0]
                matrix = np.array([
                    [coefficients[0], coefficients[1], coefficients[2]],
                    [coefficients[3], coefficients[4], coefficients[5]]
                ])
                
                # Ignore flips and unrealistic area changes.
                determinant = np.linalg.det(matrix[:2, :2])
                if determinant <= 0.002 or determinant > 20:
                    continue
                
                # Count point pairs that agree with this transform.
                source_points_homogeneous = np.hstack([source_points, np.ones((point_count, 1))])
                predicted_destination_points = source_points_homogeneous @ matrix.T
                errors = np.linalg.norm(destination_points - predicted_destination_points, axis=1)
                inliers = errors < threshold
                inlier_count = np.sum(inliers)
                
                if inlier_count > best_count:
                    best_count = inlier_count
                    best_matrix = matrix
                    best_inliers = inliers
                    print(f"        Iteration {iteration}: {inlier_count} inliers, det={determinant:.3f}")
            
            except Exception as e:
                continue
        
        # Refit using all inliers for a steadier bounding box.
        if best_matrix is not None and best_count >= 3:
            inlier_source_points = source_points[best_inliers]
            inlier_destination_points = destination_points[best_inliers]
            design = []
            target = []
            for (x, y), (u, v) in zip(inlier_source_points, inlier_destination_points):
                design.extend([[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]])
                target.extend([u, v])
            coefficients = np.linalg.lstsq(np.asarray(design), np.asarray(target), rcond=None)[0]
            best_matrix = np.array([[coefficients[0], coefficients[1], coefficients[2]],
                                    [coefficients[3], coefficients[4], coefficients[5]]])
            source_points_homogeneous = np.hstack([source_points, np.ones((point_count, 1))])
            errors = np.linalg.norm(destination_points - source_points_homogeneous @ best_matrix.T, axis=1)
            best_inliers = errors < threshold
            best_count = int(np.sum(best_inliers))

        print(f"        Best: {best_count} inliers")
        return best_matrix, best_inliers
    
    def verify_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Check candidate locations with an affine RANSAC fit.
        
        Args:
            candidates: Candidate locations from Hough voting.
            
        Returns:
            Detections that passed the checks.
        """
        print("\n[3] Verifying detections with RANSAC...")
        
        detections = []
        template_h, template_w = self.template.shape[:2]

        # Test only the ten candidates with the most votes.
        for candidate_index, candidate in enumerate(candidates[:10]):
            matches = candidate['matches']
            
            # Collect matching template and scene points.
            source_points = []
            destination_points = []
            
            for template_index, query_index in matches:
                if template_index < len(self.template_keypoints) and query_index < len(self.query_keypoints):
                    source_points.append(self.template_keypoints[template_index].pt)
                    destination_points.append(self.query_keypoints[query_index].pt)
            
            if len(source_points) < 3:
                print(f"  Candidate {candidate_index}: Skipped, only {len(source_points)} points")
                continue
            
            source_points = np.array(source_points)
            destination_points = np.array(destination_points)
            
            print(f"  Candidate {candidate_index}: Testing with {len(source_points)} points...")
            
            # Fit an affine transform.
            affine_matrix, inlier_mask = self.fit_affine_with_ransac(source_points, destination_points)
            
            if affine_matrix is None:
                print(f"  Candidate {candidate_index}: RANSAC returned None")
                continue
            
            if inlier_mask is None:
                print(f"  Candidate {candidate_index}: Inlier mask is None")
                continue
            
            inlier_count = np.sum(inlier_mask)
            print(f"  Candidate {candidate_index}: {inlier_count} inliers")

            # Four matches give one extra check beyond the three-point fit.
            if inlier_count < 4:
                continue

            # Reject shapes that are too stretched or sheared to be a packet.
            linear = affine_matrix[:, :2]
            singular_values = np.linalg.svd(linear, compute_uv=False)
            if singular_values[1] <= 1e-6 or singular_values[0] / singular_values[1] > 1.75:
                print(f"  Candidate {candidate_index}: rejected (excessive affine shear)")
                continue

            # Map the template corners into the scene.
            corners = np.array([
                [0, 0],
                [template_w, 0],
                [template_w, template_h],
                [0, template_h]
            ], dtype=np.float32)

            corners_h = np.hstack([corners, np.ones((4, 1))])
            projected = corners_h @ affine_matrix.T

            # Reject a box that is too small to be a real packet.
            signed_area = 0.5 * abs(np.sum(projected[:, 0] * np.roll(projected[:, 1], -1)
                                           - projected[:, 1] * np.roll(projected[:, 0], -1)))
            if signed_area < 400:
                print(f"  Candidate {candidate_index}: rejected (box too small)")
                continue

            # Edge packets may be partly outside the image.
            query_h, query_w = self.query.shape[:2]
            inside_x = np.logical_and(projected[:, 0] >= 0, projected[:, 0] <= query_w)
            inside_y = np.logical_and(projected[:, 1] >= 0, projected[:, 1] <= query_h)
            corners_inside = np.sum(np.logical_and(inside_x, inside_y))

            # Two visible corners are enough; otherwise check image overlap.
            if corners_inside >= 2:
                pass
            else:
                # Get the straight bounding box around the projected corners.
                box_min = np.min(projected, axis=0)
                box_max = np.max(projected, axis=0)

                # Find its overlap with the image.
                inter_min = np.maximum(box_min, [0, 0])
                inter_max = np.minimum(box_max, [query_w, query_h])
                inter_wh = inter_max - inter_min
                inter_wh = np.maximum(inter_wh, 0)
                intersection = inter_wh[0] * inter_wh[1]

                box_area = max(1.0, (box_max[0] - box_min[0]) * (box_max[1] - box_min[1]))
                overlap_ratio = intersection / box_area

                # Keep a partly visible object only when enough is in the image.
                if overlap_ratio < 0.12:
                    continue
                if inlier_count < 3:
                    continue
            
            detection = {
                'box': projected,
                'affine': affine_matrix,
                'inliers': inlier_count,
                'scale': candidate['scale'],
                'angle': candidate['angle']
            }
            
            detections.append(detection)
            print(f"  Detection {len(detections)}: {inlier_count} inliers, scale={candidate['scale']:.2f}x")
        
        return detections
    
    def remove_duplicate_detections(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Remove overlapping duplicate detections.
        
        Args:
            detections: Detections to compare.
            iou_threshold: Maximum allowed overlap.
            
        Returns:
            Detections left after removing duplicates.
        """
        print(f"\n[4] Applying Non-Maximum Suppression...")
        print(f"Detections before NMS: {len(detections)}")
        
        if len(detections) == 0:
            return detections
        
        detections = sorted(detections, key=lambda detection: detection['inliers'], reverse=True)
        kept = []
        
        while len(detections) > 0:
            current = detections[0]
            kept.append(current)
            detections = detections[1:]
            
            remaining = []
            for detection in detections:
                # Compare simple straight bounding boxes.
                box1_min = np.min(current['box'], axis=0)
                box1_max = np.max(current['box'], axis=0)
                box2_min = np.min(detection['box'], axis=0)
                box2_max = np.max(detection['box'], axis=0)
                
                inter_min = np.maximum(box1_min, box2_min)
                inter_max = np.minimum(box1_max, box2_max)
                inter = np.maximum(0, inter_max - inter_min)
                intersection = np.prod(inter)
                
                area1 = np.prod(box1_max - box1_min)
                area2 = np.prod(box2_max - box2_min)
                union = area1 + area2 - intersection
                
                iou = intersection / (union + 1e-6)
                
                if iou < iou_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        print(f"Detections after NMS: {len(kept)}")
        return kept
    
    def detect_objects(self):
        """Run the full object-detection process."""
        print("\n" + "="*60)
        print("MULTI-INSTANCE OBJECT RECOGNITION PIPELINE")
        print("="*60)

        all_detections = []

        # Run the process for every template image.
        for template_index, template_data in enumerate(self.templates):
            print(f"\nProcessing template {template_index+1}: {template_data['path']}")
            self.template = template_data['img']
            self.template_gray = template_data['gray']
            self.template_keypoints = template_data['keypoints']
            self.template_descriptors = template_data['descriptors']
            # Match features, group votes, and verify the groups.
            matches = self.find_feature_matches()
            if len(matches) < 3:
                print("Not enough matches for this template")
                continue

            candidates = self.find_hough_candidates(matches)
            if len(candidates) == 0:
                print("No candidates found for this template")
                continue

            template_detections = self.verify_candidates(candidates)
            # Save which template produced each detection.
            for detection in template_detections:
                detection['template_index'] = template_index
                detection['template_path'] = template_data['path']
            all_detections.extend(template_detections)

        if len(all_detections) == 0:
            print("No detections across templates")
            return

        # Remove weak results from every template.
        filtered_detections = self.filter_detections(all_detections)

        # Add unmatched yellow-and-red packet regions as a backup check.
        filtered_detections.extend(self.find_color_fallback_detections(filtered_detections))

        # Remove final duplicate boxes.
        self.detections = self.remove_duplicate_detections(filtered_detections)

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Total objects detected: {len(self.detections)}")
        for detection_number, detection in enumerate(self.detections, 1):
            print(f"  Object {detection_number}: {detection['inliers']} inliers, scale={detection['scale']:.2f}x, angle={detection['angle']:.1f}°")
        print("="*60)

    def find_color_fallback_detections(self, detections: List[Dict]) -> List[Dict]:
        """Find unmatched yellow-and-red packet-like regions."""
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

            # Do not add a color box that already has a SIFT detection.
            candidate_min, candidate_max = np.min(box, axis=0), np.max(box, axis=0)
            duplicate = False
            for detection in detections:
                detection_min, detection_max = np.min(detection['box'], axis=0), np.max(detection['box'], axis=0)
                inter = np.maximum(0, np.minimum(candidate_max, detection_max) - np.maximum(candidate_min, detection_min))
                intersection = float(np.prod(inter))
                union = float(np.prod(candidate_max - candidate_min) + np.prod(detection_max - detection_min) - intersection)
                if intersection / max(union, 1.0) >= 0.20:
                    duplicate = True
                    break
            if not duplicate:
                added.append({'box': box, 'inliers': 0, 'scale': 0.0, 'angle': 0.0,
                              'source': 'yellow_red_fallback'})
        if added:
            print(f"Colour fallback: added {len(added)} unmatched packet candidate(s)")
        return added

    def filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Remove weak false matches while keeping valid edge detections.

        Rules:
        - Keep detections with inliers >= 5
        - If inliers == 4, require overlap_ratio >= 0.18
        - If inliers == 3, require overlap_ratio >= 0.35
        - Remove results with a very unusual scale.
        """
        if not detections:
            return detections

        scales = np.array([detection['scale'] for detection in detections])
        median_scale = np.median(scales)

        kept = []
        for detection in detections:
            if detection.get('source') == 'yellow_red_fallback':
                kept.append(detection)
                continue
            inliers = detection['inliers']
            box = detection['box']
            box_min = np.min(box, axis=0)
            box_max = np.max(box, axis=0)
            query_h, query_w = self.query.shape[:2]
            inter_min = np.maximum(box_min, [0, 0])
            inter_max = np.minimum(box_max, [query_w, query_h])
            inter_wh = np.maximum(inter_max - inter_min, 0)
            intersection = inter_wh[0] * inter_wh[1]
            box_area = max(1.0, (box_max[0] - box_min[0]) * (box_max[1] - box_min[1]))
            overlap_ratio = intersection / box_area

            # Reject unusual sizes unless many points support them.
            if detection['scale'] > median_scale * 2.5 or detection['scale'] < median_scale / 2.5:
                if inliers < 10:
                    continue

            if inliers >= 5:
                kept.append(detection)
                continue

            if inliers == 4 and overlap_ratio >= 0.18:
                kept.append(detection)
                continue

            if inliers == 3 and overlap_ratio >= 0.35:
                kept.append(detection)
                continue

        print(f"Post-filter: {len(kept)} kept of {len(detections)}")
        return kept
    
    def visualize_results(self, output_path: str = None):
        """Save an image with detection boxes."""
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
    """Run the pipeline for every image in query_images and save the results."""
    workspace = Path(__file__).resolve().parent

    cropped = [workspace / 'Maggi_cropped.jpeg', workspace / 'Maggi_back_cropped.jpeg']
    originals = [workspace / 'Maggi.jpeg', workspace / 'Maggi_back.jpeg']
    templates = [template_path for template_path in cropped if template_path.exists()] or [template_path for template_path in originals if template_path.exists()]

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
    for query_path in query_files:
        print(f'Processing {query_path.name} ...')
        try:
            pipeline = ObjectRecognitionPipeline([str(template_path) for template_path in templates], str(query_path))
            pipeline.detect_objects()
            output_image_path = vizdir / f'results_{query_path.stem}.png'
            pipeline.visualize_results(str(output_image_path))

            detection_summary = []
            for detection in pipeline.detections:
                detection_summary.append({
                    'inliers': int(detection.get('inliers', 0)),
                    'scale': float(detection.get('scale', 0.0)),
                    'angle': float(detection.get('angle', 0.0)),
                    'box': [[float(x), float(y)] for x, y in detection['box'].tolist()]
                })

            summary.append({
                'image': query_path.name,
                'num_detections': len(pipeline.detections),
                'detections': detection_summary,
                'output_image': str(output_image_path)
            })

        except Exception as error:
            print(f'Error processing {query_path.name}:', error)
            summary.append({'image': query_path.name, 'error': str(error)})

    summary_path = outdir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nBatch run complete. Summary saved to', summary_path)
    print('Per-image visualizations saved in', vizdir)


if __name__ == "__main__":
    main()
