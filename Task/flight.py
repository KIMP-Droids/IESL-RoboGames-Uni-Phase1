#!/usr/bin/env python3
"""
Flight controller for drone competition.

Task:
1. Take off to stable height below 3m
2. Follow yellow line on dark background
3. Detect AprilTag, print tag number, turn 90° right
4. Repeat: follow line, detect tag, print, turn
5. After 3rd AprilTag: print number and land

Usage:
    python flight.py [--viz] [--connection UDP_STRING]
"""

import cv2
import time
import argparse
import signal
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

# Import our modules
from camera import SimCamera
from vision import LineDetector, AprilTagDetectorWrapper, PIDController
from drone_control import DroneController, DroneConfig


# ========================== Configuration ==========================

@dataclass
class FlightConfig:
    """Configuration for the flight mission."""
    # Connection
    connection_string: str = 'udp:0.0.0.0:14550'
    camera_host: str = 'localhost'
    camera_port: int = 5599
    
    # Flight parameters
    takeoff_altitude: float = 1.5      # meters (below 3m requirement)
    cruise_velocity: float = 0.3       # m/s forward speed while following line
    max_lateral_velocity: float = 0.4  # m/s max lateral correction
    
    # Line following PID
    pid_kp: float = 0.6
    pid_ki: float = 0.05
    pid_kd: float = 0.15
    
    # Vision parameters
    line_threshold: int = 150          # Brightness threshold for line detection
    line_min_area: int = 300           # Minimum contour area
    
    # AprilTag
    tag_family: str = 'tag36h11'
    required_tags: int = 2            # Number of tags to detect before landing
    tag_confirm_frames: int = 5        # Frames to confirm tag detection
    
    # Timing
    loop_hz: float = 20.0              # Main loop frequency
    turn_settle_time: float = 1.0      # Time to settle after turn
    tag_hover_time: float = 1.5        # Time to hover when tag detected
    
    # Safety
    max_runtime_s: int = 300           # Maximum mission time
    max_lost_line_frames: int = 30     # Frames before searching
    
    # Debug
    verbose: bool = True
    show_viz: bool = False


# ========================== State Machine ==========================

class FlightState(Enum):
    """Flight state machine states."""
    INIT = "INIT"
    TAKEOFF = "TAKEOFF"
    FOLLOW_LINE = "FOLLOW_LINE"
    TAG_DETECTED = "TAG_DETECTED"
    TURN_RIGHT = "TURN_RIGHT"
    SEARCH_LINE = "SEARCH_LINE"
    LAND = "LAND"
    COMPLETE = "COMPLETE"
    ABORT = "ABORT"


class FlightMission:
    """
    Main flight mission controller.
    Manages state machine for line following and AprilTag detection.
    """
    
    def __init__(self, config: FlightConfig):
        """Initialize flight mission."""
        self.config = config
        self.state = FlightState.INIT
        
        # Components
        self.drone: Optional[DroneController] = None
        self.camera: Optional[SimCamera] = None
        self.line_detector: Optional[LineDetector] = None
        self.tag_detector: Optional[AprilTagDetectorWrapper] = None
        self.pid_lateral: Optional[PIDController] = None
        
        # Mission state
        self.tags_detected: List[int] = []
        self.tag_confirm_count: int = 0
        self.current_tag_id: int = -1
        self.lost_line_count: int = 0
        self.start_time: float = 0
        
        # Control flag
        self._stop_requested = False
        
        # Visualization
        self._last_frame = None
        self._debug_frame = None
    
    def initialize(self) -> bool:
        """Initialize all components."""
        if self.config.verbose:
            print("=" * 50)
            print("  DRONE COMPETITION - LINE FOLLOWING MISSION")
            print("=" * 50)
            print(f"Target: Detect {self.config.required_tags} AprilTags, then land")
            print()
        
        # Initialize camera
        if self.config.verbose:
            print("[Init] Starting camera client...")
        self.camera = SimCamera(
            host=self.config.camera_host,
            port=self.config.camera_port,
            verbose=self.config.verbose
        )
        
        if not self.camera.wait_for_connection(timeout=30.0):
            print("[Init] ERROR: Camera connection timeout")
            return False
        
        # Initialize vision
        if self.config.verbose:
            print("[Init] Initializing vision modules...")
        
        self.line_detector = LineDetector(
            threshold=self.config.line_threshold,
            min_area=self.config.line_min_area,
            verbose=False
        )
        
        self.tag_detector = AprilTagDetectorWrapper(
            tag_family=self.config.tag_family,
            verbose=self.config.verbose
        )
        
        # Initialize PID controller for lateral movement
        self.pid_lateral = PIDController(
            kp=self.config.pid_kp,
            ki=self.config.pid_ki,
            kd=self.config.pid_kd,
            output_limit=self.config.max_lateral_velocity
        )
        
        # Initialize drone controller
        if self.config.verbose:
            print("[Init] Connecting to drone...")
        
        drone_config = DroneConfig(
            connection_string=self.config.connection_string,
            takeoff_altitude=self.config.takeoff_altitude,
            max_velocity_xy=self.config.max_lateral_velocity,
            verbose=self.config.verbose
        )
        
        self.drone = DroneController(drone_config)
        
        if not self.drone.connect():
            print("[Init] ERROR: Drone connection failed")
            return False
        
        if self.config.verbose:
            print("[Init] All systems initialized")
            print()
        
        return True
    
    def run(self) -> bool:
        """
        Run the flight mission.
        
        Returns:
            True if mission completed successfully
        """
        self.start_time = time.time()
        period = 1.0 / self.config.loop_hz
        
        if self.config.verbose:
            print("[Mission] Starting flight mission...")
        
        try:
            while not self._stop_requested:
                loop_start = time.time()
                
                # Check timeout
                elapsed = time.time() - self.start_time
                if elapsed > self.config.max_runtime_s:
                    print(f"[Mission] Timeout after {elapsed:.1f}s")
                    self.state = FlightState.ABORT
                
                # Get camera frame
                frame = self.camera.get_frame()
                if frame is not None:
                    self._last_frame = frame
                
                # Execute current state
                self._execute_state()
                
                # Check for mission complete
                if self.state == FlightState.COMPLETE:
                    if self.config.verbose:
                        print("[Mission] Mission completed successfully!")
                    return True
                
                if self.state == FlightState.ABORT:
                    print("[Mission] Mission aborted")
                    return False
                
                # Visualization
                if self.config.show_viz and self._debug_frame is not None:
                    cv2.imshow("Flight Debug", self._debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._stop_requested = True
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
                    
        except KeyboardInterrupt:
            print("\n[Mission] Interrupted by user")
            self.state = FlightState.ABORT
        
        finally:
            self.cleanup()
        
        return False
    
    def _execute_state(self):
        """Execute the current state."""
        if self.state == FlightState.INIT:
            self._state_init()
        elif self.state == FlightState.TAKEOFF:
            self._state_takeoff()
        elif self.state == FlightState.FOLLOW_LINE:
            self._state_follow_line()
        elif self.state == FlightState.TAG_DETECTED:
            self._state_tag_detected()
        elif self.state == FlightState.TURN_RIGHT:
            self._state_turn_right()
        elif self.state == FlightState.SEARCH_LINE:
            self._state_search_line()
        elif self.state == FlightState.LAND:
            self._state_land()
    
    def _state_init(self):
        """INIT state: Prepare for takeoff."""
        self.state = FlightState.TAKEOFF
        if self.config.verbose:
            print("[State] INIT -> TAKEOFF")
    
    def _state_takeoff(self):
        """TAKEOFF state: Arm and take off."""
        if self.drone.arm_and_takeoff(self.config.takeoff_altitude):
            if self.config.verbose:
                print(f"[State] TAKEOFF complete at {self.drone.get_altitude():.2f}m")
            
            # Give time to stabilize
            time.sleep(1.0)
            
            self.state = FlightState.FOLLOW_LINE
            if self.config.verbose:
                print("[State] TAKEOFF -> FOLLOW_LINE")
        else:
            print("[State] Takeoff failed!")
            self.state = FlightState.ABORT
    
    def _state_follow_line(self):
        """FOLLOW_LINE state: Follow the yellow line and look for AprilTags."""
        if self._last_frame is None:
            return
        
        frame = self._last_frame
        
        # Check for AprilTag first
        tags = self.tag_detector.detect(frame)
        # Filter out tags that have already been detected
        tags = [t for t in tags if t.tag_id not in self.tags_detected]
        if tags:
            tag = tags[0]  # Take first new (undetected) tag
            
            # Confirm detection over multiple frames
            if tag.tag_id == self.current_tag_id:
                self.tag_confirm_count += 1
            else:
                self.current_tag_id = tag.tag_id
                self.tag_confirm_count = 1
            
            if self.tag_confirm_count >= self.config.tag_confirm_frames:
                # Tag confirmed!
                self.drone.hover()
                self.state = FlightState.TAG_DETECTED
                if self.config.verbose:
                    print(f"[State] FOLLOW_LINE -> TAG_DETECTED (ID: {tag.tag_id})")
                return
        else:
            self.tag_confirm_count = 0
            self.current_tag_id = -1
        
        # Detect line
        line = self.line_detector.detect(frame)
        
        if line.detected:
            self.lost_line_count = 0
            
            # Calculate lateral correction using PID
            lateral_correction = self.pid_lateral.compute(line.offset_x)
            
            # Apply velocities:
            # vx = forward (cruise speed)
            # vy = lateral correction (positive = right in body frame)
            vx = self.config.cruise_velocity
            vy = lateral_correction
            
            self.drone.send_velocity_body(vx, vy, 0)
            
            if self.config.verbose and int(time.time() * 2) % 2 == 0:
                print(f"[Line] offset={line.offset_x:+.2f} correction={lateral_correction:+.2f} "
                      f"vx={vx:.2f} vy={vy:+.2f}")
        else:
            self.lost_line_count += 1
            
            if self.lost_line_count > self.config.max_lost_line_frames:
                self.drone.hover()
                self.state = FlightState.SEARCH_LINE
                if self.config.verbose:
                    print("[State] FOLLOW_LINE -> SEARCH_LINE (line lost)")
            else:
                # Continue forward slowly while searching
                self.drone.send_velocity_body(self.config.cruise_velocity * 0.3, 0, 0)
        
        # Update debug frame
        if self.config.show_viz:
            self._debug_frame = self.line_detector.draw_debug(frame, line)
            if tags:
                self._debug_frame = self.tag_detector.draw_debug(self._debug_frame, tags)
            self._draw_status(self._debug_frame)
    
    def _state_tag_detected(self):
        """TAG_DETECTED state: Hover, print tag, then turn."""
        # Hover in place
        self.drone.hover()
        
        # Add to detected list if not already there
        if self.current_tag_id not in self.tags_detected:
            self.tags_detected.append(self.current_tag_id)
            
            print()
            print("=" * 40)
            print(f"  APRILTAG DETECTED: ID = {self.current_tag_id}")
            print(f"  Tags found: {len(self.tags_detected)}/{self.config.required_tags}")
            print("=" * 40)
            print()
        
        # Wait a moment to confirm detection
        time.sleep(self.config.tag_hover_time)
        
        # Check if we have enough tags
        if len(self.tags_detected) >= self.config.required_tags:
            self.state = FlightState.LAND
            if self.config.verbose:
                print("[State] TAG_DETECTED -> LAND (all tags found!)")
        else:
            self.state = FlightState.TURN_RIGHT
            if self.config.verbose:
                print("[State] TAG_DETECTED -> TURN_RIGHT")
    
    def _state_turn_right(self):
        """TURN_RIGHT state: Turn 90 degrees clockwise."""
        if self.config.verbose:
            print("[Turn] Turning 90° right...")
        
        # Reset PID for clean start after turn
        self.pid_lateral.reset()
        
        # Execute turn
        self.drone.turn_right_90(blocking=True)
        
        # Settle time
        self.drone.hover()
        time.sleep(self.config.turn_settle_time)
        
        # Reset counters
        self.lost_line_count = 0
        self.tag_confirm_count = 0
        self.current_tag_id = -1
        
        self.state = FlightState.FOLLOW_LINE
        if self.config.verbose:
            print(f"[State] TURN_RIGHT -> FOLLOW_LINE (heading: {self.drone.get_heading():.1f}°)")
    
    def _state_search_line(self):
        """SEARCH_LINE state: Search for the line if lost."""
        if self._last_frame is None:
            return
        
        frame = self._last_frame
        line = self.line_detector.detect(frame)
        
        if line.detected:
            self.lost_line_count = 0
            self.state = FlightState.FOLLOW_LINE
            if self.config.verbose:
                print("[State] SEARCH_LINE -> FOLLOW_LINE (line found)")
            return
        
        # Rotate slowly to search for line
        self.drone.send_velocity_body(0, 0, 0, yaw_rate=10)  # Slow rotation
        
        # Timeout - just continue forward
        if self.lost_line_count > self.config.max_lost_line_frames * 3:
            self.drone.send_velocity_body(self.config.cruise_velocity * 0.5, 0, 0)
            self.lost_line_count = 0
            self.state = FlightState.FOLLOW_LINE
            if self.config.verbose:
                print("[State] SEARCH_LINE -> FOLLOW_LINE (timeout, continuing)")
    
    def _state_land(self):
        """LAND state: Land the drone."""
        if self.config.verbose:
            print()
            print("=" * 50)
            print("  MISSION COMPLETE - LANDING")
            print(f"  Tags detected: {self.tags_detected}")
            print("=" * 50)
            print()
        
        self.drone.land()
        self.drone.wait_for_landing(timeout=30.0)
        
        self.state = FlightState.COMPLETE
    
    def _draw_status(self, frame):
        """Draw mission status on debug frame."""
        h, w = frame.shape[:2]
        
        # Draw state
        state_text = f"State: {self.state.value}"
        cv2.putText(frame, state_text, (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw tags
        tags_text = f"Tags: {len(self.tags_detected)}/{self.config.required_tags} {self.tags_detected}"
        cv2.putText(frame, tags_text, (10, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw altitude
        alt_text = f"Alt: {self.drone.get_altitude():.2f}m"
        cv2.putText(frame, alt_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def stop(self):
        """Request mission stop."""
        self._stop_requested = True
    
    def cleanup(self):
        """Clean up resources."""
        if self.config.verbose:
            print("[Cleanup] Shutting down...")
        
        # Stop drone movement
        if self.drone and self.drone._connected:
            try:
                self.drone.hover()
                # If not landed, land now
                if self.drone.is_armed():
                    print("[Cleanup] Emergency landing...")
                    self.drone.land()
            except:
                pass
            self.drone.disconnect()
        
        # Stop camera
        if self.camera:
            self.camera.stop()
        
        # Close visualization
        if self.config.show_viz:
            cv2.destroyAllWindows()
        
        if self.config.verbose:
            print("[Cleanup] Complete")
            print()
            print("=" * 50)
            print("  MISSION SUMMARY")
            print(f"  Duration: {time.time() - self.start_time:.1f}s")
            print(f"  Tags detected: {self.tags_detected}")
            print(f"  Final state: {self.state.value}")
            print("=" * 50)


# ========================== Main ==========================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Drone Line Following Mission')
    parser.add_argument('--connection', type=str, default='udp:0.0.0.0:14550',
                       help='MAVLink connection string')
    parser.add_argument('--camera-host', type=str, default='localhost',
                       help='Camera stream host')
    parser.add_argument('--camera-port', type=int, default=5599,
                       help='Camera stream port')
    parser.add_argument('--viz', action='store_true',
                       help='Show debug visualization')
    parser.add_argument('--altitude', type=float, default=1.5,
                       help='Takeoff altitude in meters')
    parser.add_argument('--threshold', type=int, default=150,
                       help='Line detection threshold (0-255)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output')
    args = parser.parse_args()
    
    # Build configuration
    config = FlightConfig(
        connection_string=args.connection,
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        takeoff_altitude=args.altitude,
        line_threshold=args.threshold,
        show_viz=args.viz,
        verbose=not args.quiet
    )
    
    # Create mission
    mission = FlightMission(config)
    
    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        print("\n[Main] Interrupt received, stopping mission...")
        mission.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize
    if not mission.initialize():
        print("[Main] Initialization failed!")
        return 1
    
    # Run mission
    success = mission.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
