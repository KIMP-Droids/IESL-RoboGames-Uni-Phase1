#!/usr/bin/env python3
import cv2
import time
import math
import argparse
import threading
import signal
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict

from dronekit import connect, VehicleMode
from pymavlink import mavutil

from ultralytics import YOLO
import numpy as np
from picamera2 import Picamera2

# ---------------------------- Config ----------------------------
@dataclass
class Config:
    # Pi hardware defaults; override with --connection/--baud if you prefer
    connection_str: str = '/dev/ttyAMA0'
    baud: int = 57600
    takeoff_alt_m: float = 3.0

    loop_hz: float = 10.0
    max_vxy_mps: float = 0.7
    descend_rate_mps: float = 0.25
    hold_descend_rate_mps: float = 0.15

    center_px_tol: int = 20
    min_detect_conf: float = 0.6
    stable_frames_required: int = 6
    land_alt_trigger_m: float = 0.12
    bbox_area_land_ratio_letter: float = 0.02
    max_lost_frames: int = 20
    max_runtime_s: int = 180

    # Arena & search
    arena_length: float = 25.0       # along-X distance (m)
    arena_width: float  = 25.0         # total width (m)
    search_margin_m: float = 1.0       # safety margin from borders (m)
    search_speed_mps: float = 1.2      # lane cruise speed
    search_lane_overlap: float = 0.20  # overlap (0.2 => 20%)
    fov_footprint_x_at_3m: float = 3.0
    fov_footprint_y_at_3m: float = 2.0
    shift_y_sign: int = +1             # +1 => shift "right" in body Y each lane

    # Verbosity
    verbose: bool = True

    yolo_model_path: str = "yolo11n_ncnn_model"
    yolo_imgsz: int = 320
    letter_allowlist: Optional[List[str]] = tuple("ABCDEFG")

CFG = Config()

# ------------------------- Utilities ---------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def wrap_deg_360(a):
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def send_body_velocity_yaw(vehicle, vx, vy, vz, yaw_deg):
    # Body-frame velocity, yaw setpoint (no yaw rate)
    type_mask = 0b0000101111000111
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        int(type_mask),
        0, 0, 0,
        float(vx), float(vy), float(vz),
        0, 0, 0,
        math.radians(float(yaw_deg)),
        0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# ----------------------- Picamera2 capture ---------------------
class CameraNode:
    def __init__(self, width=640, height=480, fps=30, verbose=True):
        self.frame = None
        self.frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._fps = float(fps)
        self.verbose = verbose
        self._last_print = 0.0

        self._picam2 = Picamera2()
        cfg = self._picam2.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self._picam2.configure(cfg)
        self._picam2.start()

        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        if self.verbose:
            print(f"[Camera] Picamera2 started at {width}x{height}@{fps}fps")

    def _loop(self):
        period = 1.0 / self._fps if self._fps > 0 else 0.033
        while not self._stop.is_set():
            try:
                rgb = self._picam2.capture_array()  # (H,W,3) RGB888
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                with self.frame_lock:
                    self.frame = bgr
                if self.verbose and time.time() - self._last_print > 1.0:
                    h, w = bgr.shape[:2]
                    print(f"[Camera] New frame: {w}x{h}")
                    self._last_print = time.time()
            except Exception as e:
                if self.verbose:
                    print(f"[Camera] capture error: {e}")
            time.sleep(period)

    def get_latest_frame(self):
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self._stop.set()
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._picam2.stop()
        except Exception:
            pass
        if self.verbose:
            print("[Camera] Stopped")

# -------------------------- Detector ---------------------------
@dataclass
class Detection:
    ok: bool
    conf: float = 0.0
    letter: Optional[str] = None
    center: Tuple[int, int] = (0, 0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    bbox_area_ratio: float = 0.0

class YoloLetterDetector:
    def __init__(self, model_path: str, imgsz: int, min_conf: float,
                 allowlist: Optional[set] = None, debug: bool = False, verbose: bool = True):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.min_conf = float(min_conf)
        self.allowlist = set([s.upper() for s in allowlist]) if allowlist else None
        self.debug = debug
        self.verbose = verbose
        self._last_log = 0.0
        self.names: Union[dict, list] = self.model.names

    def set_allowlist(self, allow):
        self.allowlist = None if allow is None else set(s.upper() for s in allow)
        if self.verbose:
            print(f"[YOLO] allowlist set to: {sorted(list(self.allowlist)) if self.allowlist else 'NONE'}")

    def _class_name(self, idx: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names.get(idx, str(idx)))
        elif isinstance(self.names, list):
            if 0 <= idx < len(self.names):
                return str(self.names[idx])
            return str(idx)
        return str(idx)

    def infer(self, frame_bgr: np.ndarray) -> Detection:
        try:
            h, w = frame_bgr.shape[:2]
            area = float(h * w)
            results = self.model(frame_bgr, imgsz=self.imgsz, verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                if self.debug and self.verbose and time.time() - self._last_log > 0.5:
                    print("[YOLO] no boxes")
                    self._last_log = time.time()
                return Detection(ok=False)

            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            best = None
            best_score = -1.0
            for i in range(len(xyxy)):
                conf = float(confs[i])
                if conf < self.min_conf:
                    continue
                label_raw = self._class_name(int(clss[i]))
                label_norm = label_raw.strip().upper()
                if self.allowlist and label_norm not in self.allowlist:
                    continue

                x1, y1, x2, y2 = map(int, xyxy[i])
                x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
                bw = max(1, x2 - x1); bh = max(1, y2 - y1)
                area_ratio = float(bw * bh) / area
                score = conf * (1.0 + 2.0 * area_ratio)

                if score > best_score:
                    best = Detection(
                        ok=True, conf=conf, letter=label_norm,
                        center=(x1 + bw // 2, y1 + bh // 2),
                        bbox=(x1, y1, bw, bh),
                        bbox_area_ratio=area_ratio
                    )
                    best_score = score

            if self.verbose and best:
                x, y, bw, bh = best.bbox
                print(f"[YOLO] BEST letter={best.letter} conf={best.conf:.3f} "
                      f"center={best.center} bbox=({x},{y},{bw},{bh}) area_ratio={best.bbox_area_ratio:.4f}")
            return best if best else Detection(ok=False)
        except Exception as e:
            if self.verbose:
                print(f"[YOLO] infer error: {e}")
            return Detection(ok=False)

# --------------------- Precision Lander ------------------------
class PrecisionLander(threading.Thread):
    """
    Same yawless boustrophedon as your Gazebo version:
      RUN_X (vx = ±speed) until lane_length_m or target found
      SHIFT_Y (vy = +speed*0.8 * shift_sign) by lane_step_m
      flip forward dir and repeat
    """
    def __init__(self, vehicle, cfg: Config, cam_node: "CameraNode",
                 detector: YoloLetterDetector, viz: bool,
                 resume_state: Optional[Dict] = None, verbose: bool = True):
        super().__init__(daemon=True)
        self.vehicle = vehicle
        self.cfg = cfg
        self.cam_node = cam_node
        self.frame_lock = cam_node.frame_lock
        self.detector = detector
        self.viz = viz
        self.verbose = verbose

        self.cx_ref = 320
        self.cy_ref = 240
        self.frame_w = 640
        self.frame_h = 480

        self.state = "SEARCH"
        self.stable_frames = 0
        self.lost_frames = 0
        self.last_detect: Optional[Detection] = None
        self.start_time = time.time()

        try:
            self.yaw_target_deg = float(self.vehicle.heading or 0.0)
        except Exception:
            self.yaw_target_deg = 0.0

        self._stop = threading.Event()
        self._last_debug_frame = None

        # lane_step = max(0.5, cfg.fov_footprint_y_at_3m * (1.0 - cfg.search_lane_overlap))
        lane_step = 4
        lane_len  = max(2.0, cfg.arena_length - 2.0 * cfg.search_margin_m)
        width_lim = max(1.0, cfg.arena_width  - 2.0 * cfg.search_margin_m)
        self.max_lanes = int(width_lim // lane_step) + 1

        self.search = {
            "mode": "RUN_X",            # RUN_X or SHIFT_Y
            "lane_idx": 0,
            "forward": True,            # +X when True, -X when False
            "lane_step_m": lane_step,
            "lane_length_m": lane_len,
            "shift_sign": 1 if cfg.shift_y_sign >= 0 else -1,
            "along_m": 0.0,
            "shift_m": 0.0,
            "last_step_t": time.time(),
        }
        if resume_state:
            for k in ("mode","lane_idx","forward","along_m","shift_m"):
                if k in resume_state:
                    self.search[k] = resume_state[k]

        if self.verbose:
            print(f"[SEARCH:init] lane_step={lane_step:.2f}m lane_len={lane_len:.2f}m "
                  f"width_lim={width_lim:.2f}m max_lanes={self.max_lanes} "
                  f"resume={resume_state is not None}")

    # ---------- logging helpers ----------
    def _t(self) -> str:
        return f"{(time.time()-self.start_time):7.2f}s"

    def _log(self, msg: str):
        if self.verbose:
            s = self.search
            print(f"[{self._t()}][{self.state}] lane={s['lane_idx']} mode={s['mode']} "
                  f"fwd={s['forward']} along={s['along_m']:.2f} shift={s['shift_m']:.2f} :: {msg}")

    # ---------- helpers ----------
    def _flush_stop(self, pause_s=0.08):
        self._log("FLUSH stop (vx=0, vy=0, vz=0)")
        send_body_velocity_yaw(self.vehicle, 0.0, 0.0, 0.0, 0)
        time.sleep(pause_s)

    def _drive(self, vx: float, vy: float, vz: float = 0.0):
        self._log(f"DRIVE cmd vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f} yaw={self.yaw_target_deg:.1f}°")
        send_body_velocity_yaw(self.vehicle, vx, vy, vz, 0)

    def _get_latest_frame(self):
        frame = self.cam_node.get_latest_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if (w, h) != (self.frame_w, self.frame_h):
            if self.verbose:
                print(f"[{self._t()}][Video] resize detected {self.frame_w}x{self.frame_h} -> {w}x{h}")
            self.frame_w, self.frame_h = w, h
            self.cx_ref = w // 2
            self.cy_ref = h // 2
        return frame

    # ------------------------ thread loop -------------------------
    def run(self):
        period = 1.0 / self.cfg.loop_hz
        self._log(f"Thread start, loop_hz={self.cfg.loop_hz}")
        while not self._stop.is_set():
            if time.time() - self.start_time > self.cfg.max_runtime_s:
                print("[Lander] Max runtime exceeded, stopping.")
                break

            frame = self._get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            det = self.detector.infer(frame)
            self.last_detect = det

            if self.state == "SEARCH":
                self.search_step(det)
            elif self.state == "ALIGN":
                self.align_step(det)
            elif self.state == "DESCEND":
                self.descend_step(det)
            elif self.state == "LAND":
                self.land_step()
                break
            elif self.state == "ABORT":
                self.abort_step()
            else:
                self._log("Unknown state -> SEARCH")
                self.state = "SEARCH"

            dbg = frame.copy()
            self.draw_debug(dbg, det)
            self._last_debug_frame = dbg

            time.sleep(max(0.0, period))

        try:
            self._flush_stop()
        except Exception as e:
            print(f"[{self._t()}] flush_stop error: {e}")

    # -------------------- SEARCH (yawless lanes) -------------------
    def search_step(self, det: Detection):
        if det.ok and det.conf >= self.cfg.min_detect_conf:
            self._log(f"DETECTED letter={det.letter} conf={det.conf:.3f} -> ALIGN")
            self.state = "ALIGN"
            self.stable_frames = 0
            self._flush_stop()
            return

        now = time.time()
        dt = max(0.0, now - self.search["last_step_t"])
        self.search["last_step_t"] = now

        spd = float(self.cfg.search_speed_mps)  # preserve original tested direction
        lane_len  = self.search["lane_length_m"]
        lane_step = self.search["lane_step_m"]
        mode = self.search["mode"]

        if mode == "RUN_X":
            vx = spd if self.search["forward"] else -spd
            self.search["along_m"] += abs(vx) * dt
            self._drive(vx, 0.0, 0.0)

            if self.search["along_m"] >= lane_len:
                self._log(f"Lane end reached (along={self.search['along_m']:.2f}/{lane_len:.2f}) -> SHIFT_Y")
                self.search["mode"] = "SHIFT_Y"
                self.search["along_m"] = 0.0
                self.search["shift_m"] = 0.0
                self._flush_stop()
            return

        if mode == "SHIFT_Y":
            vy = self.search["shift_sign"] * spd * 0.8
            self.search["shift_m"] += abs(vy) * dt
            self._drive(0.0, -vy, 0.0)

            if self.search["shift_m"] >= lane_step:
                self.search["lane_idx"] += 1
                self._log(f"Shift complete (shift={self.search['shift_m']:.2f}/{lane_step:.2f}). "
                          f"lane_idx={self.search['lane_idx']}/{self.max_lanes-1}")
                if self.search["lane_idx"] >= self.max_lanes:
                    print("[Lander] SEARCH coverage complete -> ABORT (re-evaluate)")
                    self._flush_stop()
                    self.state = "ABORT"
                    return

                self.search["mode"] = "RUN_X"
                self.search["forward"] = not self.search["forward"]
                self.search["along_m"] = 0.0
                self._flush_stop()
            return

        # Fallback safety
        self._log("Fallback to RUN_X")
        self.search["mode"] = "RUN_X"
        self._flush_stop()

    # ---------------- ALIGN / DESCEND / LAND / ABORT --------------
    def align_step(self, det: Detection):
        if not (det.ok and det.conf >= self.cfg.min_detect_conf):
            self.lost_frames += 1
            self._log(f"ALIGN: lost frame {self.lost_frames}/{self.cfg.max_lost_frames}")
            if self.lost_frames > self.cfg.max_lost_frames:
                self._log("ALIGN: too many lost -> ABORT")
                self.state = "ABORT"
                self.lost_frames = 0
            else:
                self._flush_stop()
            return

        self.lost_frames = 0
        ex = det.center[0] - self.cx_ref
        ey = det.center[1] - self.cy_ref
        exn = ex / (self.frame_w * 0.5)
        eyn = (self.cy_ref - det.center[1]) / (self.frame_h * 0.5)

        kp_xy = 0.8
        vx = clamp(kp_xy * eyn * self.cfg.max_vxy_mps, -self.cfg.max_vxy_mps, self.cfg.max_vxy_mps)
        vy = clamp(kp_xy * exn * self.cfg.max_vxy_mps, -self.cfg.max_vxy_mps, self.cfg.max_vxy_mps)

        self._log(f"ALIGN det=({det.letter},{det.conf:.2f}) px_err=({ex},{ey}) "
                  f"cmd=(-vx,-vy)=({-vx:+.3f},{-vy:+.3f})")
        self._drive(-vx, -vy, 0.0)

        if abs(ex) <= self.cfg.center_px_tol and abs(ey) <= self.cfg.center_px_tol:
            self.stable_frames += 1
            self._log(f"ALIGN center OK ({self.stable_frames}/{self.cfg.stable_frames_required})")
            if self.stable_frames >= self.cfg.stable_frames_required:
                self._log("-> DESCEND")
                self.state = "DESCEND"
                self.stable_frames = 0
        else:
            self.stable_frames = 0

    def descend_step(self, det: Detection):
        alt = float(self.vehicle.location.global_relative_frame.alt or 0.0)

        if not (det.ok and det.conf >= self.cfg.min_detect_conf):
            self.lost_frames += 1
            self._log(f"DESCEND: lost frame {self.lost_frames}/{self.cfg.max_lost_frames}")
            self._flush_stop()
            if self.lost_frames > self.cfg.max_lost_frames:
                self._log("DESCEND: too many lost -> ABORT")
                self.state = "ABORT"
                self.lost_frames = 0
            return

        self.lost_frames = 0

        ex = det.center[0] - self.cx_ref
        ey = det.center[1] - self.cy_ref
        exn = ex / (self.frame_w * 0.5)
        eyn = (self.cy_ref - det.center[1]) / (self.frame_h * 0.5)

        kp_align = 0.7
        vx = clamp(kp_align * eyn * self.cfg.max_vxy_mps, -self.cfg.max_vxy_mps, self.cfg.max_vxy_mps)
        vy = clamp(kp_align * exn * self.cfg.max_vxy_mps, -self.cfg.max_vxy_mps, self.cfg.max_vxy_mps)

        center_ok = (abs(ex) <= self.cfg.center_px_tol and abs(ey) <= self.cfg.center_px_tol)
        vz = self.cfg.hold_descend_rate_mps if not center_ok else self.cfg.descend_rate_mps

        self._log(f"DESCEND det=({det.letter},{det.conf:.2f}) px_err=({ex},{ey}) "
                  f"cmd=(vx,vy,vz)=({vx:+.3f},{vy:+.3f},{vz:+.3f}) alt={alt:.2f}m "
                  f"area_ratio={det.bbox_area_ratio:.4f} center_ok={center_ok}")
        self._drive(vx, vy, vz)

        if det.bbox_area_ratio >= self.cfg.bbox_area_land_ratio_letter or alt <= self.cfg.land_alt_trigger_m:
            self._log("Close enough -> LAND")
            self.state = "LAND"

    def land_step(self):
        self._log("Switching to LAND mode")
        self.vehicle.mode = VehicleMode("LAND")

    def abort_step(self):
        self._log("ABORT: brief climb + stop, then re-search")
        self._flush_stop()
        self._drive(0.0, 0.0, -0.2)
        time.sleep(1.2)
        self._flush_stop()
        self.search.update({"mode": "RUN_X", "along_m": 0.0, "shift_m": 0.0})
        self.state = "SEARCH"

    # ------------------------ Debug overlay -----------------------
    def draw_debug(self, frame, det: Detection):
        cv2.drawMarker(frame, (self.cx_ref, self.cy_ref), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        if det and det.ok:
            x, y, bw, bh = det.bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cx, cy = det.center
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            txt = f"{det.letter or '?'} conf={det.conf:.2f} area={det.bbox_area_ratio:.3f} state={self.state}"
        else:
            txt = f"no target state={self.state}"
        cv2.putText(frame, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if self.state == "SEARCH":
            s = self.search
            info = f"lane={s['lane_idx']} {s['mode']} fwd={s['forward']} along={s['along_m']:.1f}m shift={s['shift_m']:.1f}m step={s['lane_step_m']:.1f} len={s['lane_length_m']:.1f}"
            cv2.putText(frame, info, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)

    def get_coverage_state(self) -> Dict:
        st = {
            "mode": self.search["mode"],
            "lane_idx": self.search["lane_idx"],
            "forward": self.search["forward"],
            "along_m": self.search["along_m"],
            "shift_m": self.search["shift_m"],
        }
        if self.verbose:
            print(f"[{self._t()}][SEARCH:save] {st}")
        return st

    def stop(self):
        self._log("Thread stop requested")
        self._stop.set()

# --------------------------- Basics ----------------------------
def wait_ready_to_arm(vehicle):
    print("[PreArm] Waiting for vehicle to initialise...")
    while not vehicle.is_armable:
        time.sleep(0.5)
    try:
        print(f"[PreArm] GPS fix={vehicle.gps_0.fix_type}, EKF ok={vehicle.ekf_ok}")
    except Exception as e:
        print(f"[PreArm] Status fetch error: {e}")

def status_thread(vehicle, period=2.0):
    def loop():
        try:
            alt = float(vehicle.location.global_relative_frame.alt or 0.0)
            print(f"[Status] Bat={vehicle.battery} Mode={vehicle.mode.name} Alt={alt:.2f} m")
        except Exception as e:
            print(f"[Status] error: {e}")
        t = threading.Timer(period, loop)
        t.daemon = True
        t.start()
    loop()

def arm_and_takeoff(vehicle, target_alt):
    wait_ready_to_arm(vehicle)
    print("[Arming]")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print("  waiting for arming...")
        time.sleep(0.5)
    print(f"[Takeoff] to {target_alt:.1f} m")
    vehicle.simple_takeoff(target_alt)
    while True:
        alt = float(vehicle.location.global_relative_frame.alt or 0.0)
        print(f"  Altitude: {alt:.2f} m")
        if alt >= target_alt - 0.5:
            print("[Takeoff] reached")
            break
        time.sleep(0.5)

# -------- one leg: seek a specific letter, then land -----------
def fly_to_letter(vehicle, cam_node, detector, target_letter, viz=False,
                  coverage_state: Optional[Dict] = None,
                  max_runtime_s: Optional[int] = None, verbose: bool = True) -> Dict:
    detector.set_allowlist({target_letter})
    print(f"[Mission] Seeking letter '{target_letter}' "
          f"(resume={'yes' if coverage_state else 'no'})")

    lander = PrecisionLander(vehicle, CFG, cam_node, detector, viz=viz,
                             resume_state=coverage_state, verbose=verbose)
    lander.start()

    t0 = time.time()
    try:
        if viz:
            try:
                cv2.namedWindow("precision_landing_demo", cv2.WINDOW_NORMAL)
                print("[Viz] Opened window 'precision_landing_demo'")
            except Exception as e:
                print(f"[Viz] Window error: {e}")

        while True:
            if viz and lander._last_debug_frame is not None:
                cv2.imshow("precision_landing_demo", lander._last_debug_frame)
                cv2.waitKey(1)

            # end condition (some FCs disarm after LAND completes)
            if vehicle.mode.name == "LAND" and not vehicle.armed:
                print(f"[Mission] Landed on '{target_letter}' and disarmed")
                break

            if max_runtime_s is None:
                max_runtime_s = CFG.max_runtime_s
            if time.time() - t0 > max_runtime_s:
                print("[Mission] Timeout -> switching to LAND")
                vehicle.mode = VehicleMode("LAND")
                break

            time.sleep(0.02 if viz else 0.5)
    finally:
        lander.stop()
        st = lander.get_coverage_state()
        try:
            send_body_velocity_yaw(vehicle, 0.0, 0.0, 0.0, 0)
            print("[Mission] Stopped vehicle (vx=vy=vz=0)")
        except Exception as e:
            print(f"[Mission] stop error: {e}")
        if viz:
            try:
                cv2.destroyWindow("precision_landing_demo")
                for _ in range(3):
                    cv2.waitKey(1)
                print("[Viz] Window closed")
            except Exception as e:
                print(f"[Viz] Close error: {e}")
    return st

# ------------------------------ Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true", help="Show debug window")
    ap.add_argument("--model", type=str, default=CFG.yolo_model_path)
    ap.add_argument("--imgsz", type=int, default=CFG.yolo_imgsz)
    ap.add_argument("--conf", type=float, default=CFG.min_detect_conf)
    ap.add_argument("--targets", type=str, default="A,B,C", help="Comma-separated letters")
    ap.add_argument("--connection", type=str, default=CFG.connection_str)
    ap.add_argument("--baud", type=int, default=CFG.baud)
    ap.add_argument("--camera-width", type=int, default=640)
    ap.add_argument("--camera-height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--debug-det", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Reduce printing")
    args = ap.parse_args()

    CFG.verbose = not args.quiet
    CFG.yolo_imgsz = args.imgsz
    CFG.min_detect_conf = args.conf

    targets = ["A" , "B", "C", "D"]
    if not targets:
        raise SystemExit("No targets provided. Use --targets A,B,...")

    print(f"[Args] viz={args.viz} targets={targets} imgsz={args.imgsz} conf={args.conf} "
          f"model='{args.model}' camera={args.camera_width}x{args.camera_height}@{args.camera_fps} "
          f"conn='{args.connection}' baud={args.baud} verbose={CFG.verbose} debug_det={args.debug_det}")

    print("[Connect] Connecting to vehicle...")
    vehicle = connect(args.connection, baud=args.baud)
    print("[Connect] OK!")
    status_thread(vehicle)

    # camera
    cam_node = CameraNode(width=args.camera_width, height=args.camera_height, fps=args.camera_fps, verbose=CFG.verbose)

    detector = YoloLetterDetector(
        model_path=args.model, imgsz=args.imgsz, min_conf=args.conf,
        allowlist=None, debug=args.debug_det, verbose=CFG.verbose
    )
    print(f"[YOLO] model.names = {detector.names}")

    # Ctrl-C friendly
    stop_now = {"sig": False}
    def _sigint(*_):
        stop_now["sig"] = True
        print("\n[Main] SIGINT received — wrapping up…")
    signal.signal(signal.SIGINT, _sigint)

    coverage_state = None  # persisted across legs
    try:
        for i, tgt in enumerate(targets):
            if stop_now["sig"]:
                break
            print(f"[Main] === Target {i+1}/{len(targets)}: {tgt} ===")
            arm_and_takeoff(vehicle, CFG.takeoff_alt_m)

            coverage_state = fly_to_letter(vehicle, cam_node, detector, tgt,
                                           viz=args.viz, coverage_state=coverage_state,
                                           verbose=CFG.verbose)
            if i < len(targets) - 1:
                print("[Main] Preparing for next target...")
            else:
                print("[Main] Sequence complete")
    finally:
        try:
            send_body_velocity_yaw(vehicle, 0.0, 0.0, 0.0, 0.0)
            print("[Main] Final stop command sent")
        except Exception as e:
            print(f"[Main] stop error: {e}")
        try:
            vehicle.close()
            print("[Main] Vehicle link closed")
        except Exception as e:
            print(f"[Main] vehicle close error: {e}")
        try:
            cam_node.stop()
        except Exception as e:
            print(f"[Main] camera close error: {e}")

if __name__ == "__main__":
    main()
