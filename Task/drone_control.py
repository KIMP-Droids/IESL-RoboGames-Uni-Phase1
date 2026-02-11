#!/usr/bin/env python3
"""
Drone control module using DroneKit for ArduPilot SITL communication.
Provides high-level functions for takeoff, movement, yaw control, and landing.
"""

import time
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil


@dataclass
class DroneConfig:
    """Configuration for drone control."""
    connection_string: str = 'udp:0.0.0.0:14550'
    takeoff_altitude: float = 2.5      # meters (below 3m as required)
    max_velocity_xy: float = 0.5       # m/s horizontal
    max_velocity_z: float = 0.3        # m/s vertical
    yaw_rate: float = 30.0             # degrees per second
    position_tolerance: float = 0.3    # meters
    yaw_tolerance: float = 5.0         # degrees
    arm_timeout: float = 30.0          # seconds
    takeoff_timeout: float = 30.0      # seconds
    verbose: bool = True


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


class DroneController:
    """
    High-level drone controller using DroneKit.
    Provides methods for takeoff, velocity control, yaw rotation, and landing.
    """
    
    def __init__(self, config: DroneConfig = None):
        """
        Initialize drone controller.
        
        Args:
            config: DroneConfig instance (uses defaults if None)
        """
        self.config = config or DroneConfig()
        self.vehicle = None
        self._connected = False
        self._armed = False
    
    def connect(self, connection_string: str = None) -> bool:
        """
        Connect to the drone via MAVLink.
        
        Args:
            connection_string: MAVLink connection string (e.g., 'udp:0.0.0.0:14550')
            
        Returns:
            True if connected successfully
        """
        conn_str = connection_string or self.config.connection_string
        
        if self.config.verbose:
            print(f"[Drone] Connecting to {conn_str}...")
        
        try:
            self.vehicle = connect(conn_str, wait_ready=True, timeout=60)
            self._connected = True
            
            if self.config.verbose:
                print(f"[Drone] Connected!")
                print(f"[Drone] Firmware: {self.vehicle.version}")
                print(f"[Drone] Mode: {self.vehicle.mode.name}")
                print(f"[Drone] Armed: {self.vehicle.armed}")
            
            return True
            
        except Exception as e:
            print(f"[Drone] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the drone."""
        if self.vehicle:
            try:
                self.vehicle.close()
                if self.config.verbose:
                    print("[Drone] Disconnected")
            except Exception as e:
                print(f"[Drone] Disconnect error: {e}")
        self._connected = False
        self.vehicle = None
    
    def wait_ready_to_arm(self, timeout: float = None) -> bool:
        """
        Wait for vehicle to be ready to arm.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if ready to arm
        """
        timeout = timeout or self.config.arm_timeout
        start = time.time()
        
        if self.config.verbose:
            print("[Drone] Waiting for vehicle to be armable...")
        
        while not self.vehicle.is_armable:
            if time.time() - start > timeout:
                print("[Drone] Timeout waiting for armable state")
                return False
            time.sleep(0.5)
        
        if self.config.verbose:
            print("[Drone] Vehicle is armable")
            try:
                print(f"[Drone] GPS: fix={self.vehicle.gps_0.fix_type}, satellites={self.vehicle.gps_0.satellites_visible}")
                print(f"[Drone] EKF OK: {self.vehicle.ekf_ok}")
                print(f"[Drone] Battery: {self.vehicle.battery}")
            except:
                pass
        
        return True
    
    def arm(self) -> bool:
        """
        Arm the vehicle.
        
        Returns:
            True if armed successfully
        """
        if not self._connected:
            print("[Drone] Not connected")
            return False
        
        # Set GUIDED mode
        if self.config.verbose:
            print("[Drone] Setting GUIDED mode...")
        
        self.vehicle.mode = VehicleMode("GUIDED")
        
        # Wait for mode change
        start = time.time()
        while self.vehicle.mode.name != "GUIDED":
            if time.time() - start > 10:
                print("[Drone] Failed to set GUIDED mode")
                return False
            time.sleep(0.1)
        
        if self.config.verbose:
            print("[Drone] Arming motors...")
        
        self.vehicle.armed = True
        
        # Wait for arming
        start = time.time()
        while not self.vehicle.armed:
            if time.time() - start > self.config.arm_timeout:
                print("[Drone] Timeout waiting for arming")
                return False
            if self.config.verbose:
                print("  Waiting for arming...")
            time.sleep(0.5)
        
        self._armed = True
        if self.config.verbose:
            print("[Drone] Armed!")
        
        return True
    
    def takeoff(self, altitude: float = None) -> bool:
        """
        Perform takeoff to specified altitude.
        
        Args:
            altitude: Target altitude in meters (uses config default if None)
            
        Returns:
            True if takeoff completed
        """
        target_alt = altitude or self.config.takeoff_altitude
        
        if not self._armed:
            print("[Drone] Not armed, cannot takeoff")
            return False
        
        if self.config.verbose:
            print(f"[Drone] Taking off to {target_alt}m...")
        
        self.vehicle.simple_takeoff(target_alt)
        
        # Wait for takeoff
        start = time.time()
        while True:
            current_alt = self.vehicle.location.global_relative_frame.alt or 0
            
            if self.config.verbose:
                print(f"  Altitude: {current_alt:.2f}m / {target_alt}m")
            
            if current_alt >= target_alt - 0.5:
                if self.config.verbose:
                    print(f"[Drone] Takeoff complete at {current_alt:.2f}m")
                return True
            
            if time.time() - start > self.config.takeoff_timeout:
                print("[Drone] Takeoff timeout")
                return False
            
            time.sleep(0.5)
    
    def arm_and_takeoff(self, altitude: float = None) -> bool:
        """
        Complete arm and takeoff sequence.
        
        Args:
            altitude: Target altitude in meters
            
        Returns:
            True if successful
        """
        if not self.wait_ready_to_arm():
            return False
        
        if not self.arm():
            return False
        
        return self.takeoff(altitude)
    
    def send_velocity_body(self, vx: float, vy: float, vz: float, yaw_rate: float = 0):
        """
        Send velocity command in body frame.
        
        Args:
            vx: Forward velocity (m/s, positive = forward)
            vy: Right velocity (m/s, positive = right)
            vz: Down velocity (m/s, positive = down)
            yaw_rate: Yaw rate (deg/s, positive = clockwise)
        """
        if not self._connected:
            return
        
        # Clamp velocities
        vx = clamp(vx, -self.config.max_velocity_xy, self.config.max_velocity_xy)
        vy = clamp(vy, -self.config.max_velocity_xy, self.config.max_velocity_xy)
        vz = clamp(vz, -self.config.max_velocity_z, self.config.max_velocity_z)
        
        # Type mask: ignore position, use velocity and yaw rate
        # Bits: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, ax, ay, az, yaw, yaw_rate
        # We want to control vel_x, vel_y, vel_z, yaw_rate
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )
        
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,       # time_boot_ms
            0, 0,    # target system, component
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0, 0, 0,  # position (ignored)
            vx, vy, vz,  # velocity
            0, 0, 0,  # acceleration (ignored)
            0,        # yaw (ignored)
            math.radians(yaw_rate)  # yaw_rate
        )
        
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
    
    def send_velocity_ned(self, vn: float, ve: float, vd: float):
        """
        Send velocity command in NED (North-East-Down) frame.
        
        Args:
            vn: North velocity (m/s)
            ve: East velocity (m/s)
            vd: Down velocity (m/s)
        """
        if not self._connected:
            return
        
        # Clamp velocities
        vn = clamp(vn, -self.config.max_velocity_xy, self.config.max_velocity_xy)
        ve = clamp(ve, -self.config.max_velocity_xy, self.config.max_velocity_xy)
        vd = clamp(vd, -self.config.max_velocity_z, self.config.max_velocity_z)
        
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,       # time_boot_ms
            0, 0,    # target system, component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # type_mask (only velocity)
            0, 0, 0,  # position
            vn, ve, vd,  # velocity
            0, 0, 0,  # acceleration
            0, 0      # yaw, yaw_rate
        )
        
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
    
    def hover(self):
        """Stop all movement and hover in place."""
        self.send_velocity_body(0, 0, 0)
    
    def rotate_yaw(self, angle_deg: float, relative: bool = True, 
                   direction: int = 1, blocking: bool = True) -> bool:
        """
        Rotate the drone's yaw.
        
        Args:
            angle_deg: Yaw angle in degrees
            relative: If True, rotate relative to current heading
            direction: 1 for clockwise, -1 for counter-clockwise
            blocking: If True, wait for rotation to complete
            
        Returns:
            True if rotation completed (or started if non-blocking)
        """
        if not self._connected:
            return False
        
        if self.config.verbose:
            print(f"[Drone] Rotating {'CW' if direction > 0 else 'CCW'} {angle_deg}°")
        
        # Use condition_yaw command
        # direction: 1 = CW, -1 = CCW
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,    # target system, component
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,       # confirmation
            abs(angle_deg),   # param1: target angle
            self.config.yaw_rate,  # param2: yaw speed deg/s
            direction,  # param3: direction (1=CW, -1=CCW)
            1 if relative else 0,  # param4: relative (1) or absolute (0)
            0, 0, 0  # params 5-7 (unused)
        )
        
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        
        if not blocking:
            return True
        
        # Wait for rotation to complete
        rotation_time = abs(angle_deg) / self.config.yaw_rate
        time.sleep(rotation_time + 0.5)  # Add buffer time
        
        if self.config.verbose:
            print(f"[Drone] Rotation complete, heading: {self.get_heading()}°")
        
        return True
    
    def turn_right_90(self, blocking: bool = True) -> bool:
        """
        Turn 90 degrees to the right (clockwise).
        
        Args:
            blocking: Wait for turn to complete
            
        Returns:
            True if turn completed/started
        """
        return self.rotate_yaw(90, relative=True, direction=1, blocking=blocking)
    
    def land(self) -> bool:
        """
        Switch to LAND mode.
        
        Returns:
            True if land mode set
        """
        if not self._connected:
            return False
        
        if self.config.verbose:
            print("[Drone] Landing...")
        
        self.vehicle.mode = VehicleMode("LAND")
        
        # Wait for mode change
        start = time.time()
        while self.vehicle.mode.name != "LAND":
            if time.time() - start > 5:
                print("[Drone] Failed to set LAND mode")
                return False
            time.sleep(0.1)
        
        return True
    
    def wait_for_landing(self, timeout: float = 60.0) -> bool:
        """
        Wait for the drone to land and disarm.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if landed and disarmed
        """
        start = time.time()
        
        while time.time() - start < timeout:
            alt = self.vehicle.location.global_relative_frame.alt or 0
            
            if self.config.verbose:
                print(f"  Landing altitude: {alt:.2f}m")
            
            if alt < 0.2 and not self.vehicle.armed:
                if self.config.verbose:
                    print("[Drone] Landed and disarmed")
                self._armed = False
                return True
            
            time.sleep(0.5)
        
        print("[Drone] Landing timeout")
        return False
    
    def get_altitude(self) -> float:
        """Get current altitude in meters."""
        if not self._connected:
            return 0.0
        return self.vehicle.location.global_relative_frame.alt or 0.0
    
    def get_heading(self) -> float:
        """Get current heading in degrees (0-360)."""
        if not self._connected:
            return 0.0
        return self.vehicle.heading or 0.0
    
    def get_mode(self) -> str:
        """Get current flight mode."""
        if not self._connected:
            return "UNKNOWN"
        return self.vehicle.mode.name
    
    def is_armed(self) -> bool:
        """Check if vehicle is armed."""
        if not self._connected:
            return False
        return self.vehicle.armed
    
    def get_battery(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get battery status.
        
        Returns:
            Tuple of (voltage, level_percent)
        """
        if not self._connected:
            return (None, None)
        
        bat = self.vehicle.battery
        return (bat.voltage, bat.level)
    
    def print_status(self):
        """Print current vehicle status."""
        if not self._connected:
            print("[Drone] Not connected")
            return
        
        print(f"[Drone Status]")
        print(f"  Mode: {self.get_mode()}")
        print(f"  Armed: {self.is_armed()}")
        print(f"  Altitude: {self.get_altitude():.2f}m")
        print(f"  Heading: {self.get_heading():.1f}°")
        voltage, level = self.get_battery()
        print(f"  Battery: {voltage:.2f}V ({level}%)" if voltage else "  Battery: N/A")


if __name__ == "__main__":
    # Test the drone controller
    print("Drone control module loaded")
    
    config = DroneConfig(
        connection_string='udp:0.0.0.0:14550',
        verbose=True
    )
    
    controller = DroneController(config)
    
    print("\nTo test, run with simulation:")
    print("  python drone_control.py")
    print("\nThis will connect and print status (no flying).")
