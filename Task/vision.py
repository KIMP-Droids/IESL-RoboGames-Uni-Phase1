#!/usr/bin/env python3
"""
Vision module for line following and AprilTag detection.
Uses OpenCV for line detection and pupil-apriltags for AprilTag detection.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time

# Try to import apriltag library
try:
    from pupil_apriltags import Detector as AprilTagDetector
    APRILTAG_AVAILABLE = True
except ImportError:
    try:
        import apriltag
        APRILTAG_AVAILABLE = True
    except ImportError:
        APRILTAG_AVAILABLE = False
        print("[Vision] Warning: AprilTag library not found. Install with: pip install pupil-apriltags")


@dataclass
class LineDetection:
    """Result of line detection."""
    detected: bool
    center_x: int = 0          # X position of line center in image
    center_y: int = 0          # Y position of line center in image
    angle: float = 0.0         # Angle of line in degrees (-90 to 90)
    offset_x: float = 0.0      # Normalized offset from image center (-1 to 1)
    confidence: float = 0.0    # Detection confidence (0 to 1)
    contour: np.ndarray = None # The detected contour


@dataclass  
class AprilTagResult:
    """Result of AprilTag detection."""
    detected: bool
    tag_id: int = -1
    center: Tuple[int, int] = (0, 0)
    corners: List[Tuple[int, int]] = None
    confidence: float = 0.0


class PIDController:
    """
    Simple PID controller for smooth line following.
    """
    
    def __init__(self, kp: float = 0.5, ki: float = 0.0, kd: float = 0.1,
                 output_limit: float = 1.0):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
            output_limit: Maximum absolute output value
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
    
    def reset(self):
        """Reset the PID controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
    
    def compute(self, error: float, dt: float = None) -> float:
        """
        Compute PID output.
        
        Args:
            error: Current error value
            dt: Time delta (if None, uses wall clock)
            
        Returns:
            Control output (clamped to output_limit)
        """
        now = time.time()
        
        if dt is None:
            if self._prev_time is None:
                dt = 0.1
            else:
                dt = now - self._prev_time
        
        dt = max(dt, 0.001)  # Prevent division by zero
        
        # Proportional
        p_term = self.kp * error
        
        # Integral (with anti-windup)
        self._integral += error * dt
        self._integral = max(-self.output_limit / max(self.ki, 0.001), 
                            min(self.output_limit / max(self.ki, 0.001), self._integral))
        i_term = self.ki * self._integral
        
        # Derivative
        derivative = (error - self._prev_error) / dt
        d_term = self.kd * derivative
        
        # Update state
        self._prev_error = error
        self._prev_time = now
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Clamp output
        return max(-self.output_limit, min(self.output_limit, output))


class LineDetector:
    """
    Detects yellow/bright lines on dark background using grayscale images.
    Since camera provides grayscale, we detect based on brightness.
    """
    
    def __init__(self, 
                 threshold: int = 180,
                 min_area: int = 500,
                 roi_top_fraction: float = 0.2,
                 roi_bottom_fraction: float = 0.9,
                 verbose: bool = False):
        """
        Initialize line detector.
        
        Args:
            threshold: Brightness threshold for line detection (0-255)
            min_area: Minimum contour area to be considered a line
            roi_top_fraction: Top of ROI as fraction of image height
            roi_bottom_fraction: Bottom of ROI as fraction of image height
            verbose: Enable debug printing
        """
        self.threshold = threshold
        self.min_area = min_area
        self.roi_top_fraction = roi_top_fraction
        self.roi_bottom_fraction = roi_bottom_fraction
        self.verbose = verbose
        
        # Morphological kernel for noise removal
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    def detect(self, frame: np.ndarray) -> LineDetection:
        """
        Detect bright line in grayscale image.
        
        Args:
            frame: Grayscale image (height, width)
            
        Returns:
            LineDetection result
        """
        if frame is None or len(frame.shape) != 2:
            return LineDetection(detected=False)
        
        height, width = frame.shape
        
        # Define ROI (region of interest) - focus on middle portion
        roi_top = int(height * self.roi_top_fraction)
        roi_bottom = int(height * self.roi_bottom_fraction)
        roi = frame[roi_top:roi_bottom, :]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Threshold to get bright regions (yellow line on dark background)
        _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return LineDetection(detected=False)
        
        # Find the largest contour (should be the line)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < self.min_area:
            return LineDetection(detected=False)
        
        # Get centroid
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return LineDetection(detected=False)
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) + roi_top  # Adjust for ROI offset
        
        # Fit line to get angle
        angle = 0.0
        if len(largest) >= 5:
            try:
                # Fit line using least squares
                vx, vy, x, y = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = np.degrees(np.arctan2(vy, vx))
                # Normalize angle to -90 to 90
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
            except:
                pass
        
        # Calculate normalized offset from center
        offset_x = (cx - width / 2) / (width / 2)
        
        # Confidence based on area and position
        max_expected_area = width * (roi_bottom - roi_top) * 0.3
        confidence = min(1.0, area / max_expected_area)
        
        if self.verbose:
            print(f"[Line] cx={cx}, cy={cy}, angle={angle:.1f}°, offset={offset_x:.2f}, conf={confidence:.2f}")
        
        return LineDetection(
            detected=True,
            center_x=cx,
            center_y=cy,
            angle=angle,
            offset_x=offset_x,
            confidence=confidence,
            contour=largest
        )
    
    def draw_debug(self, frame: np.ndarray, detection: LineDetection) -> np.ndarray:
        """
        Draw debug visualization on frame.
        
        Args:
            frame: Grayscale or BGR image
            detection: Line detection result
            
        Returns:
            BGR image with debug overlay
        """
        # Convert to BGR if grayscale
        if len(frame.shape) == 2:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis = frame.copy()
        
        height, width = vis.shape[:2]
        
        # Draw center line
        cv2.line(vis, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
        
        # Draw ROI boundaries
        roi_top = int(height * self.roi_top_fraction)
        roi_bottom = int(height * self.roi_bottom_fraction)
        cv2.line(vis, (0, roi_top), (width, roi_top), (255, 255, 0), 1)
        cv2.line(vis, (0, roi_bottom), (width, roi_bottom), (255, 255, 0), 1)
        
        if detection.detected:
            # Draw contour
            if detection.contour is not None:
                # Adjust contour for ROI offset
                adjusted = detection.contour.copy()
                adjusted[:, :, 1] += roi_top
                cv2.drawContours(vis, [adjusted], -1, (0, 0, 255), 2)
            
            # Draw centroid
            cv2.circle(vis, (detection.center_x, detection.center_y), 10, (255, 0, 0), -1)
            
            # Draw angle indicator
            length = 50
            angle_rad = np.radians(detection.angle)
            end_x = int(detection.center_x + length * np.cos(angle_rad))
            end_y = int(detection.center_y + length * np.sin(angle_rad))
            cv2.line(vis, (detection.center_x, detection.center_y), (end_x, end_y), (0, 255, 255), 2)
            
            # Draw info text
            info = f"Offset: {detection.offset_x:+.2f} Angle: {detection.angle:.1f}°"
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(vis, "No line detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis


class AprilTagDetectorWrapper:
    """
    Wrapper for AprilTag detection using pupil-apriltags or apriltag library.
    """
    
    def __init__(self, 
                 tag_family: str = "tag36h11",
                 verbose: bool = False):
        """
        Initialize AprilTag detector.
        
        Args:
            tag_family: AprilTag family (default: tag36h11)
            verbose: Enable debug printing
        """
        self.verbose = verbose
        self.detector = None
        
        if not APRILTAG_AVAILABLE:
            print("[AprilTag] WARNING: No AprilTag library available!")
            return
        
        try:
            # Try pupil-apriltags first
            from pupil_apriltags import Detector
            self.detector = Detector(
                families=tag_family,
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25
            )
            if self.verbose:
                print(f"[AprilTag] Initialized pupil-apriltags with family '{tag_family}'")
        except ImportError:
            try:
                # Fall back to apriltag
                import apriltag
                self.detector = apriltag.Detector(apriltag.DetectorOptions(families=tag_family))
                if self.verbose:
                    print(f"[AprilTag] Initialized apriltag with family '{tag_family}'")
            except Exception as e:
                print(f"[AprilTag] Failed to initialize: {e}")
    
    def detect(self, frame: np.ndarray) -> List[AprilTagResult]:
        """
        Detect AprilTags in frame.
        
        Args:
            frame: Grayscale image
            
        Returns:
            List of AprilTagResult objects
        """
        results = []
        
        if self.detector is None or frame is None:
            return results
        
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            detections = self.detector.detect(frame)
            
            for det in detections:
                # Handle both pupil-apriltags and apriltag formats
                if hasattr(det, 'tag_id'):
                    tag_id = det.tag_id
                    center = (int(det.center[0]), int(det.center[1]))
                    corners = [(int(c[0]), int(c[1])) for c in det.corners]
                    confidence = det.decision_margin if hasattr(det, 'decision_margin') else 1.0
                else:
                    # apriltag library format
                    tag_id = det.tag_id
                    center = (int(det.center[0]), int(det.center[1]))
                    corners = [(int(c[0]), int(c[1])) for c in det.corners]
                    confidence = det.decision_margin if hasattr(det, 'decision_margin') else 1.0
                
                result = AprilTagResult(
                    detected=True,
                    tag_id=tag_id,
                    center=center,
                    corners=corners,
                    confidence=confidence
                )
                results.append(result)
                
                if self.verbose:
                    print(f"[AprilTag] Detected tag ID={tag_id} at {center}")
                    
        except Exception as e:
            if self.verbose:
                print(f"[AprilTag] Detection error: {e}")
        
        return results
    
    def detect_single(self, frame: np.ndarray) -> AprilTagResult:
        """
        Detect a single AprilTag (returns the one with highest confidence).
        
        Args:
            frame: Grayscale image
            
        Returns:
            AprilTagResult (detected=False if none found)
        """
        results = self.detect(frame)
        
        if not results:
            return AprilTagResult(detected=False)
        
        # Return the one with highest confidence
        return max(results, key=lambda r: r.confidence)
    
    def draw_debug(self, frame: np.ndarray, results: List[AprilTagResult]) -> np.ndarray:
        """
        Draw debug visualization for AprilTag detections.
        
        Args:
            frame: Image to draw on
            results: List of detection results
            
        Returns:
            Image with debug overlay
        """
        if len(frame.shape) == 2:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis = frame.copy()
        
        for result in results:
            if not result.detected:
                continue
            
            # Draw corners and lines
            if result.corners:
                pts = np.array(result.corners, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            
            # Draw center
            cv2.circle(vis, result.center, 5, (0, 0, 255), -1)
            
            # Draw ID
            text = f"ID: {result.tag_id}"
            cv2.putText(vis, text, (result.center[0] + 10, result.center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis


if __name__ == "__main__":
    # Test the vision module with a test image
    print("Vision module loaded successfully")
    print(f"AprilTag library available: {APRILTAG_AVAILABLE}")
    
    # Create test image with a bright line
    test_frame = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(test_frame, (320, 100), (350, 400), 255, 20)
    
    detector = LineDetector(verbose=True)
    result = detector.detect(test_frame)
    print(f"Line detection test: {result}")
    
    if APRILTAG_AVAILABLE:
        tag_detector = AprilTagDetectorWrapper(verbose=True)
        print("AprilTag detector initialized")
