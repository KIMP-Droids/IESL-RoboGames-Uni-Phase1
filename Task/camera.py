#!/usr/bin/env python3
"""
Camera module for Webots drone simulation.
Connects to the camera stream via TCP socket and provides frames.
"""

import socket
import struct
import threading
import time
import numpy as np

class SimCamera:
    """
    TCP-based camera client for Webots simulation.
    Connects to the camera stream on port 5599 and provides grayscale frames.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 5599, 
                 verbose: bool = True, reconnect: bool = True):
        """
        Initialize the camera client.
        
        Args:
            host: Camera stream host address
            port: Camera stream port (default 5599)
            verbose: Enable debug printing
            reconnect: Auto-reconnect on connection loss
        """
        self.host = host
        self.port = port
        self.verbose = verbose
        self.reconnect = reconnect
        
        self.frame = None
        self.frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = False
        self._sock = None
        
        self.width = 640
        self.height = 480
        
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        
        if self.verbose:
            print(f"[Camera] Starting camera client for {host}:{port}")
    
    def _connect(self) -> bool:
        """Establish connection to the camera stream."""
        try:
            if self._sock:
                try:
                    self._sock.close()
                except:
                    pass
            
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self.host, self.port))
            self._connected = True
            
            if self.verbose:
                print(f"[Camera] Connected to {self.host}:{self.port}")
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Camera] Connection failed: {e}")
            self._connected = False
            return False
    
    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            remaining = n - len(data)
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise ConnectionResetError("Connection closed")
            data += chunk
        return data
    
    def _stream_loop(self):
        """Main loop for receiving camera frames."""
        while not self._stop.is_set():
            # Try to connect if not connected
            if not self._connected:
                if not self._connect():
                    if self.reconnect:
                        time.sleep(1.0)
                        continue
                    else:
                        break
            
            try:
                # Read header (4 bytes: width + height as unsigned shorts)
                header = self._recv_exact(4)
                width, height = struct.unpack('=HH', header)
                
                # Update dimensions if changed
                if width != self.width or height != self.height:
                    self.width = width
                    self.height = height
                    if self.verbose:
                        print(f"[Camera] Frame size: {width}x{height}")
                
                # Read image data (grayscale: width * height bytes)
                img_size = width * height
                img_data = self._recv_exact(img_size)
                
                # Convert to numpy array
                frame = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
                
                with self.frame_lock:
                    self.frame = frame
                    
            except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
                if self.verbose:
                    print(f"[Camera] Connection lost: {e}")
                self._connected = False
                if not self.reconnect:
                    break
                    
            except Exception as e:
                if self.verbose:
                    print(f"[Camera] Error: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> np.ndarray:
        """
        Get the latest camera frame.
        
        Returns:
            Grayscale numpy array (height, width) or None if no frame available
        """
        with self.frame_lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._connected
    
    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for camera connection.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self._connected and self.frame is not None:
                return True
            time.sleep(0.1)
        return False
    
    def stop(self):
        """Stop the camera client."""
        self._stop.set()
        try:
            if self._sock:
                self._sock.close()
        except:
            pass
        try:
            self._thread.join(timeout=2.0)
        except:
            pass
        if self.verbose:
            print("[Camera] Stopped")


if __name__ == "__main__":
    # Test the camera module
    import cv2
    
    cam = SimCamera(verbose=True)
    print("Waiting for camera connection...")
    
    if cam.wait_for_connection(timeout=10.0):
        print("Camera connected! Press 'q' to quit.")
        
        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("Camera Feed", frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    else:
        print("Failed to connect to camera")
    
    cam.stop()
