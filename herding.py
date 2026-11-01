"""Malchy — bring a stray sheep back to the herd with a drone.

Malchy finds the stray with the same rule as the video analysis (an animal
much farther from the herd centre than a typical one), then drives the drone:
  1. APPROACH  — fly high (20 m), where sheep ignore the drone
  2. DESCEND   — drop to 6 m behind the stray, on the far side from the herd
  3. PUSH      — bark and keep that position behind the stray, pushing it home
                 (if the drone falls out of position it climbs above 12 m and
                 comes back, so it never drives the sheep the wrong way)
  4. RETURN    — climb, fly home, land, report to the shepherd in Telegram

Sheep follow a Strömbom-style shepherding model (Strömbom et al., 2014):
they graze, keep near their neighbours and flee from a low drone; a bark
widens and strengthens that reaction.

The same planner drives two backends:
  --drone sim      built-in kinematic drone (quick preview, no autopilot)
  --drone mavlink  real ArduPilot over MAVLink: SITL in Gazebo (default
                   tcp:127.0.0.1:5762) or the Pixhawk on the drone
                   (e.g. --connect /dev/ttyTHS1 --baud 921600 on the Jetson)

Usage:
  python3 herding.py --drone sim
  python3 herding.py --drone mavlink --gazebo --telegram
Outputs: herding_topdown.mp4 (Malchy's top-down view), herding_log.json
Coordinates: metres, x = east, y = north, home (take-off point) at 0, 0.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np

ALARM_ALT = 12.0      # sheep ignore a drone flying higher than this
SAFE_ALT = 20.0       # approach / return altitude
PUSH_ALT = 6.0        # herding altitude
BEHIND = 5.0          # how far behind the stray the drone holds
SETTLE_GAP = 7.0      # near the herd: stop pushing, let the stray walk in by itself
REJOIN_DIST = 3.0     # stray is back when this close to a herd mate


# ---------------------------------------------------------------- sheep ---

class Flock:
    """Strömbom-style sheep: grazing, local cohesion, fleeing from the drone."""

    def __init__(self, positions, seed=1):
        self.p = np.array(positions, dtype=float)
        self.heading = np.zeros_like(self.p)
        self.rng = np.random.default_rng(seed)

    def step(self, dt, drone, barking):
        """drone = (x, y, alt) or None."""
        n = len(self.p)
        new_heading = np.zeros_like(self.p)
        speed = np.zeros(n)
        alert_r = 16.0 if barking else 12.0
        for i in range(n):
            diff = self.p - self.p[i]
            dist = np.hypot(diff[:, 0], diff[:, 1])
            dist[i] = np.inf
            # personal space: push away from sheep closer than 1 m
            close = dist < 1.0
            repel = -(diff[close] / dist[close, None]).sum(axis=0) if close.any() else np.zeros(2)
            noise = self.rng.normal(0, 1, 2)
            alarmed = False
            if drone is not None and drone[2] <= ALARM_ALT:
                away = self.p[i] - np.array(drone[:2])
                d_dog = math.hypot(*away)
                alarmed = d_dog < alert_r
            if alarmed:
                k = np.argsort(dist)[:8]                      # 8 nearest neighbours
                to_lcm = self.p[k].mean(axis=0) - self.p[i]
                h = (0.5 * unit(self.heading[i]) + 1.05 * unit(to_lcm) + 2.0 * unit(repel)
                     + (1.5 if barking else 1.0) * unit(away) + 0.3 * unit(noise))
                new_heading[i] = unit(h)
                speed[i] = 1.8 if barking else 1.4
            else:
                near = dist < 8.0                            # only neighbours it can see
                cohesion = np.zeros(2)
                if near.any():
                    to_group = self.p[near].mean(axis=0) - self.p[i]
                    if np.hypot(*to_group) > 2.5:
                        cohesion = unit(to_group)
                h = cohesion + 2.0 * unit(repel) + 0.6 * unit(noise)
                new_heading[i] = unit(h)
                speed[i] = 0.35 if cohesion.any() or repel.any() else 0.12
        self.heading = new_heading
        self.p += new_heading * speed[:, None] * dt


def unit(v):
    norm = math.hypot(v[0], v[1])
    return v / norm if norm > 1e-9 else np.zeros(2)


def make_scenario(seed, n_herd=24, herd=(48.0, 22.0), stray=(30.0, 46.0)):
    rng = random.Random(seed)
    pts = []
    while len(pts) < n_herd:
        a, r = rng.uniform(0, 2 * math.pi), 4.5 * math.sqrt(rng.random())
        p = (herd[0] + r * math.cos(a), herd[1] + r * math.sin(a))
        if all(math.dist(p, q) > 1.0 for q in pts):
            pts.append(p)
    pts.append(stray)
    return pts


def find_strays(points, factor=2.5, min_dist=10.0):
    """Same rule as track_herd.py: much farther from the centre than a typical animal."""
    c = np.median(points, axis=0)
    d = np.hypot(*(points - c).T)
    typical = float(np.median(d)) or 1.0
    return [int(i) for i in np.argsort(-d) if d[i] > max(factor * typical, min_dist)]


def herd_centre(points, exclude):
    mask = np.ones(len(points), bool)
    mask[exclude] = False
    return np.median(points[mask], axis=0)


# ---------------------------------------------------------------- drones ---

class SimDrone:
    """Kinematic stand-in for ArduPilot: same interface as MavlinkDrone."""

    def __init__(self, step=0.1):
        self.pos = [0.0, 0.0, 0.0]
        self.target = list(self.pos)
        self.t = 0.0
        self.step = step

    def update(self):
        self.t += self.step
        x, y, z = self.pos
        tx, ty, tz = self.target
        d = math.hypot(tx - x, ty - y)
        move = min(d, 5.0 * self.step)                     # 5 m/s horizontal
        if d > 1e-6:
            x += (tx - x) / d * move
            y += (ty - y) / d * move
        z += max(-1.5 * self.step, min(2.5 * self.step, tz - z))
        self.pos = [x, y, z]

    def now(self):
        return self.t

    def start(self, alt):
        self.target = [0.0, 0.0, alt]
        while self.pos[2] < 0.9 * alt:
            self.update()

    def goto(self, x, y, alt):
        self.target = [x, y, alt]

    def land(self):
        self.target = [self.pos[0], self.pos[1], 0.0]

    def landed(self):
        return self.pos[2] < 0.2


class MavlinkDrone:
    """ArduCopter in GUIDED mode over MAVLink (SITL or the real Pixhawk)."""

    def __init__(self, connect, baud=921600):
        from pymavlink import mavutil
        self.mu = mavutil
        print(f"[drone] connecting to {connect} ...")
        self.m = mavutil.mavlink_connection(connect, baud=baud, source_system=250)
        self.m.wait_heartbeat(timeout=60)
        self.ts, self.tc = self.m.target_system, self.m.target_component
        print(f"[drone] heartbeat from system {self.ts}")
        self.m.mav.request_data_stream_send(self.ts, self.tc, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
        self.m.mav.command_long_send(self.ts, self.tc, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                     0, 32, 100000, 0, 0, 0, 0, 0)   # LOCAL_POSITION_NED at 10 Hz
        self.pos, self.t = None, None

    def update(self):
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if kind == "LOCAL_POSITION_NED":
                self.pos = [msg.y, msg.x, -msg.z]           # NED -> east, north, up
                self.t = msg.time_boot_ms / 1000.0
            elif kind == "STATUSTEXT":
                print(f"[autopilot] {msg.text}")

    def now(self):
        return self.t

    def _wait(self, condition, timeout, what):
        end = time.time() + timeout
        while time.time() < end:
            self.update()
            if condition():
                return
            time.sleep(0.1)
        raise SystemExit(f"[drone] timed out waiting for {what}")

    def set_mode(self, mode):
        end = time.time() + 15
        while time.time() < end:
            self.m.set_mode(mode)
            t0 = time.time()
            while time.time() - t0 < 1.5:
                self.update()
                if self.m.flightmode == mode:
                    print(f"[drone] mode {mode}")
                    return
                time.sleep(0.1)
        raise SystemExit(f"[drone] could not switch to {mode}")

    def start(self, alt):
        print("[drone] waiting for a position estimate (GPS/EKF) ...")
        self._wait(lambda: self.pos is not None, 120, "position estimate")
        self.set_mode("GUIDED")
        if self.m.motors_armed() and self.pos[2] > 1.0:          # already flying: just climb
            print(f"[drone] already airborne, climbing to {alt:.0f} m")
            self.goto(self.pos[0], self.pos[1], alt)
            self._wait(lambda: self.pos[2] > 0.9 * alt, 60, "climb")
            return
        print("[drone] arming ...")
        end = time.time() + 120
        while not self.m.motors_armed():
            if time.time() > end:
                raise SystemExit("[drone] could not arm — check PreArm messages above")
            self.m.arducopter_arm()
            t0 = time.time()
            while time.time() - t0 < 3 and not self.m.motors_armed():
                self.update()
                time.sleep(0.1)
        print(f"[drone] armed, taking off to {alt:.0f} m")
        self.m.mav.command_long_send(self.ts, self.tc, self.mu.mavlink.MAV_CMD_NAV_TAKEOFF,
                                     0, 0, 0, 0, 0, 0, 0, alt)
        self._wait(lambda: self.pos[2] > 0.9 * alt, 90, "take-off")

    def goto(self, x, y, alt):
        position_only = 0b110111111000
        self.m.mav.set_position_target_local_ned_send(
            0, self.ts, self.tc, self.mu.mavlink.MAV_FRAME_LOCAL_NED, position_only,
            y, x, -alt, 0, 0, 0, 0, 0, 0, 0, 0)

    def land(self):
        self.set_mode("LAND")

    def landed(self):
        return self.pos[2] < 0.3 and not self.m.motors_armed()


# ---------------------------------------------------------------- bark ---

class Barker:
    """Plays a bark through the speaker (Jetson on the real drone, laptop in SITL)."""

    def __init__(self, sound):
        self.sound = sound
        self.player = shutil.which("aplay") or shutil.which("paplay")
        self.last = -1e9

    def tick(self, on):
        if on and time.time() - self.last > 1.2:
            self.last = time.time()
            if self.sound and self.player:
                subprocess.Popen([self.player, self.sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def synth_bark(path):
    """A synthetic two-pulse bark (no recording needed)."""
    rate = 22050
    t = np.arange(int(0.16 * rate)) / rate
    pulse = np.sin(2 * np.pi * (520 - 900 * t) * t) * np.exp(-t * 22) + 0.25 * np.random.default_rng(0).normal(0, 1, t.size) * np.exp(-t * 30)
    gap = np.zeros(int(0.09 * rate))
    audio = np.concatenate([pulse, gap, 0.8 * pulse])
    audio = (audio / np.abs(audio).max() * 0.8 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.tobytes())
    return path


# ---------------------------------------------------------------- gazebo ---

SHEEP_SDF = """<?xml version="1.0"?>
<sdf version="1.9"><model name="sheep"><static>true</static><link name="body">
<visual name="wool"><pose>0 0 0.6 0 0 0</pose><geometry><ellipsoid><radii>0.55 0.32 0.3</radii></ellipsoid></geometry>
<material><ambient>0.93 0.92 0.88 1</ambient><diffuse>0.96 0.95 0.9 1</diffuse></material></visual>
<visual name="head"><pose>0.6 0 0.78 0 0.3 0</pose><geometry><box><size>0.28 0.18 0.2</size></box></geometry>
<material><ambient>0.15 0.12 0.1 1</ambient><diffuse>0.2 0.16 0.13 1</diffuse></material></visual>
{legs}
</link></model></sdf>"""
LEG = ('<visual name="leg{i}"><pose>{x} {y} 0.18 0 0 0</pose><geometry><cylinder><radius>0.045</radius>'
       '<length>0.36</length></cylinder></geometry><material><ambient>0.15 0.12 0.1 1</ambient>'
       '<diffuse>0.2 0.16 0.13 1</diffuse></material></visual>')


class GazeboSheep:
    """Spawns sheep in Gazebo and moves them every frame (set_pose_vector)."""

    def __init__(self, world, positions):
        legs = "".join(LEG.format(i=i, x=x, y=y) for i, (x, y) in
                       enumerate([(0.3, 0.15), (0.3, -0.15), (-0.3, 0.15), (-0.3, -0.15)]))
        self.sdf = SHEEP_SDF.format(legs=legs)
        self.world = world
        self.names = [f"sheep_{i + 1}" for i in range(len(positions))]
        self.busy = False
        try:
            from gz.msgs10.boolean_pb2 import Boolean
            from gz.msgs10.entity_factory_v_pb2 import EntityFactory_V
            from gz.msgs10.pose_v_pb2 import Pose_V
            from gz.transport13 import Node
            self.node, self.Pose_V, self.Boolean = Node(), Pose_V, Boolean
            req = EntityFactory_V()
            for name, (x, y) in zip(self.names, positions):
                f = req.data.add()
                f.sdf, f.name = self.sdf, name
                f.pose.position.x, f.pose.position.y = x, y
            ok, _ = self.node.request(f"/world/{world}/create_multiple", req, EntityFactory_V, Boolean, 5000)
            self.mode = "python"
        except ImportError:
            self.mode = "cli"
            self.sdf_file = Path(tempfile.gettempdir()) / "malchy_sheep.sdf"
            self.sdf_file.write_text(self.sdf)
            items = ", ".join(f'{{sdf_filename: "{self.sdf_file}", name: "{n}", '
                              f'pose: {{position: {{x: {x:.2f}, y: {y:.2f}}}}}}}'
                              for n, (x, y) in zip(self.names, positions))
            ok = self._cli("create_multiple", "gz.msgs.EntityFactory_V", f"data: [{items}]", 8000)
        print(f"[gazebo] spawned {len(self.names)} sheep via {self.mode}: {'ok' if ok else 'FAILED'}")

    def _cli(self, service, reqtype, req, timeout=1000):
        result = subprocess.run(["gz", "service", "-s", f"/world/{self.world}/{service}", "--reqtype", reqtype,
                                 "--reptype", "gz.msgs.Boolean", "--timeout", str(timeout), "--req", req],
                                capture_output=True, text=True)
        return "true" in result.stdout

    def update(self, positions, headings):
        poses = []
        for name, (x, y), (hx, hy) in zip(self.names, positions, headings):
            yaw = math.atan2(hy, hx) if (hx or hy) else 0.0
            poses.append((name, x, y, math.sin(yaw / 2), math.cos(yaw / 2)))
        if self.mode == "python":
            req = self.Pose_V()
            for name, x, y, qz, qw in poses:
                p = req.pose.add()
                p.name = name
                p.position.x, p.position.y = x, y
                p.orientation.z, p.orientation.w = qz, qw
            self.node.request(f"/world/{self.world}/set_pose_vector", req, self.Pose_V, self.Boolean, 300)
        elif not self.busy:                                  # the gz CLI is slow: update in the background
            text = ", ".join(f'{{name: "{n}", position: {{x: {x:.2f}, y: {y:.2f}}}, '
                             f'orientation: {{z: {qz:.3f}, w: {qw:.3f}}}}}' for n, x, y, qz, qw in poses)
            self.busy = True

            def run():
                self._cli("set_pose_vector", "gz.msgs.Pose_V", f"pose: [{text}]")
                self.busy = False
            threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- view ---

class TopDown:
    """Malchy's top-down view of the pasture, written to an mp4."""

    def __init__(self, path, bounds, fps=10, size=720, panel=480):
        self.x0, self.y0, self.x1, self.y1 = bounds
        self.size, self.panel = size, panel
        self.scale = size / max(self.x1 - self.x0, self.y1 - self.y0)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size + panel, size))
        self.trail = []
        self.frame = None
        self.label = None

    def px(self, x, y):
        return int((x - self.x0) * self.scale), int(self.size - (y - self.y0) * self.scale)

    def draw(self, flock, stray, centre, drone, target, phase, barking, info, t):
        img = np.zeros((self.size, self.size + self.panel, 3), np.uint8)
        img[:, :self.size] = (60, 130, 85)                    # pasture (BGR)
        step = 10
        for gx in range(int(self.x0 // step) * step, int(self.x1) + 1, step):
            cv2.line(img, self.px(gx, self.y0), self.px(gx, self.y1), (70, 142, 96), 1)
        for gy in range(int(self.y0 // step) * step, int(self.y1) + 1, step):
            cv2.line(img, self.px(self.x0, gy), self.px(self.x1, gy), (70, 142, 96), 1)
        cv2.drawMarker(img, self.px(0, 0), (200, 200, 200), cv2.MARKER_SQUARE, 18, 2)
        cv2.putText(img, "home", (self.px(0, 0)[0] + 12, self.px(0, 0)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
        if stray is not None and phase not in ("RETURN", "LAND", "DONE"):
            cv2.line(img, self.px(*flock.p[stray]), self.px(*centre), (180, 220, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(img, self.px(*centre), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
        r = max(4, int(0.5 * self.scale))
        for i, (x, y) in enumerate(flock.p):
            colour = (0, 140, 255) if i == stray and phase not in ("RETURN", "LAND", "DONE") else (245, 245, 240)
            cv2.circle(img, self.px(x, y), r, colour, -1, cv2.LINE_AA)
            cv2.circle(img, self.px(x, y), r, (40, 40, 40), 1, cv2.LINE_AA)
        if stray is not None:
            sx, sy = self.px(*flock.p[stray])
            back = phase in ("RETURN", "LAND", "DONE")
            cv2.putText(img, f"#{self.label or stray + 1}" + (" back" if back else ""), (sx + r + 3, sy - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 230, 120) if back else (0, 140, 255), 2, cv2.LINE_AA)
        if target is not None:
            cv2.drawMarker(img, self.px(*target[:2]), (0, 255, 255), cv2.MARKER_DIAMOND, 12, 2)
        dx, dy, alt = drone
        self.trail.append(self.px(dx, dy))
        self.trail = self.trail[-3000:]
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array(self.trail)], False, (255, 210, 120), 1, cv2.LINE_AA)
        dp = self.px(dx, dy)
        low = alt <= ALARM_ALT
        dcol = (60, 60, 230) if low else (230, 160, 40)
        if low and phase in ("DESCEND", "PUSH", "SETTLE"):
            cv2.circle(img, dp, int((16 if barking else 12) * self.scale), dcol, 1, cv2.LINE_AA)
        if barking:
            ring = int((0.3 + (t * 2) % 1.0) * 6 * self.scale)
            cv2.circle(img, dp, ring, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "WOOF!", (dp[0] + 14, dp[1] - 14), cv2.FONT_HERSHEY_DUPLEX, 0.7, (80, 80, 255), 2,
                        cv2.LINE_AA)
        for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.line(img, dp, (dp[0] + ox * 9, dp[1] + oy * 9), dcol, 3, cv2.LINE_AA)
            cv2.circle(img, (dp[0] + ox * 9, dp[1] + oy * 9), 5, dcol, 1, cv2.LINE_AA)
        # side panel
        x = self.size + 24
        img[:, self.size:] = (32, 28, 26)
        cv2.putText(img, "MALCHY", (x, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "drone herding assistant", (x, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1,
                    cv2.LINE_AA)
        cv2.putText(img, phase, (x, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        for k, line in enumerate(info):
            cv2.putText(img, line, (x, 175 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
                        cv2.LINE_AA)
        legend = [((245, 245, 240), "sheep"), ((0, 140, 255), "stray"), ((230, 160, 40), "drone high (ignored)"),
                  ((60, 60, 230), "drone low (sheep react)"), ((0, 255, 255), "herding point")]
        for k, (c, label) in enumerate(legend):
            y = self.size - 150 + 26 * k
            cv2.circle(img, (x + 8, y - 5), 7, c, -1, cv2.LINE_AA)
            cv2.putText(img, label, (x + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"t = {t:5.1f} s", (x, self.size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (170, 170, 170), 1, cv2.LINE_AA)
        self.frame = img
        self.writer.write(img)
        return img

    def close(self):
        self.writer.release()


# ---------------------------------------------------------------- main ---

def toward(pos, target, max_step):
    """Carrot point at most max_step from pos toward target (limits speed near sheep)."""
    d = math.dist(pos[:2], target[:2])
    if d <= max_step:
        return target
    f = max_step / d
    return (pos[0] + (target[0] - pos[0]) * f, pos[1] + (target[1] - pos[1]) * f, target[2])


def notify(text):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from malchy_report import send_telegram
        if send_telegram(text):
            print("[sent to Telegram]")
    except Exception as error:
        print(f"[warning] Telegram: {error}")


def main():
    parser = argparse.ArgumentParser(description="Return a stray sheep to the herd with a drone")
    parser.add_argument("--drone", choices=["sim", "mavlink"], default="sim")
    parser.add_argument("--connect", default="tcp:127.0.0.1:5762", help="MAVLink address (SITL or /dev/ttyTHS1)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--gazebo", action="store_true", help="show the sheep in Gazebo")
    parser.add_argument("--world", default="iris_runway")
    parser.add_argument("--bark-sound", help="wav file to play when barking (default: synthetic bark)")
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--telegram", action="store_true", help="report to the shepherd when done")
    parser.add_argument("--show", action="store_true", help="show the top-down view live")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-time", type=float, default=300, help="give up after this many seconds")
    parser.add_argument("--label", type=int, help="number to call the stray in messages (from the report)")
    args = parser.parse_args()

    random.seed(args.seed)
    flock = Flock(make_scenario(args.seed), seed=args.seed)
    strays = find_strays(flock.p)
    if not strays:
        raise SystemExit("No stray found — the herd is together.")
    stray = strays[0]
    name = args.label or stray + 1
    print(f"[malchy] stray sheep #{stray + 1}: {math.dist(flock.p[stray], herd_centre(flock.p, stray)):.1f} m "
          f"from the herd")

    sound = None if args.no_sound else (args.bark_sound or synth_bark(str(Path(tempfile.gettempdir()) / "bark.wav")))
    barker = Barker(sound)
    gz = GazeboSheep(args.world, flock.p) if args.gazebo else None
    drone = SimDrone() if args.drone == "sim" else MavlinkDrone(args.connect, args.baud)

    pts = np.vstack([flock.p, [[0, 0]]])
    lo, hi = pts.min(axis=0) - 15, pts.max(axis=0) + 15
    span = max(hi - lo)
    mid = (lo + hi) / 2
    view = TopDown("herding_topdown.mp4", (mid[0] - span / 2, mid[1] - span / 2, mid[0] + span / 2, mid[1] + span / 2))
    view.label = name

    drone.update()
    drone.start(SAFE_ALT)
    drone.update()
    t0 = drone.now()
    t_prev = t0
    phase, events = "APPROACH", []
    start_dist = math.dist(flock.p[stray], herd_centre(flock.p, stray))
    start_centre = herd_centre(flock.p, stray)
    t_returned = None
    last_print = -10

    def log(text):
        stamp = drone.now() - t0
        events.append({"t": round(stamp, 1), "event": text})
        print(f"[{stamp:6.1f}s] {text}")

    log(f"take-off done, stray #{name} is {start_dist:.1f} m from the herd")
    try:
        while True:
            wall = time.time()
            drone.update()
            t = drone.now()
            dt = min(max(t - t_prev, 0.0), 0.5)
            t_prev = t
            pos = drone.pos
            centre = herd_centre(flock.p, stray)
            s = flock.p[stray]
            gap = float(np.min(np.hypot(*(np.delete(flock.p, stray, axis=0) - s).T)))
            away = unit(s - centre)
            behind = (s[0] + away[0] * BEHIND, s[1] + away[1] * BEHIND)
            off = math.dist(pos[:2], behind)
            to_herd = float(np.min(np.hypot(*(np.delete(flock.p, stray, axis=0) - pos[:2]).T)))
            barking = False

            if phase == "APPROACH":
                target = (*behind, SAFE_ALT)
                if off < 2.0:
                    phase = "DESCEND"
                    log("over the herding point, descending behind the stray")
            elif phase == "DESCEND":
                target = (*behind, PUSH_ALT)
                if off > 6.0:
                    phase = "APPROACH"
                elif pos[2] < PUSH_ALT + 1.0:
                    phase = "PUSH"
                    log("pushing the stray back with barks")
            elif phase == "PUSH":
                target = (*behind, PUSH_ALT)
                barking = off < 4.0 and to_herd > 17.0      # never bark at the herd itself
                if gap < SETTLE_GAP:
                    phase, hold = "SETTLE", (pos[0], pos[1], ALARM_ALT + 2)
                    log("stray is close — drone backs off so the herd stays calm")
                elif off > 7.0:
                    phase = "APPROACH"
                    log("out of position — climbing and coming round again")
            elif phase == "SETTLE":
                target = hold                               # quiet, above the alarm height
                if gap < REJOIN_DIST:
                    phase, t_returned = "RETURN", t - t0
                    log(f"stray #{name} is back with the herd")
                elif gap > SETTLE_GAP + 4:
                    phase = "APPROACH"
                    log("stray wandered off again — going round")
            elif phase == "RETURN":
                target = (pos[0], pos[1], SAFE_ALT) if pos[2] < SAFE_ALT - 1 else (0.0, 0.0, SAFE_ALT)
                if math.dist(pos[:2], (0, 0)) < 1.5 and pos[2] > SAFE_ALT - 1:
                    phase, target = "LAND", None
                    log("home, landing")
                    drone.land()
            elif phase == "LAND":
                target = None
                if drone.landed():
                    log("landed")
                    break
            if t - t0 > args.max_time:
                log("time limit reached — stopping")
                if phase not in ("LAND",):
                    drone.land()
                break

            if target is not None:
                speed = 3.0 if phase == "PUSH" else 6.0
                drone.goto(*toward(pos, target, speed * 1.0))
            flock.step(dt, pos, barking)
            barker.tick(barking)
            if gz:
                gz.update(flock.p, flock.heading)

            dist_now = math.dist(flock.p[stray], centre)
            info = [f"stray #{name}: {dist_now:5.1f} m from herd",
                    f"drone altitude: {pos[2]:4.1f} m",
                    f"bark: {'ON' if barking else 'off'}",
                    f"herd size: {len(flock.p)}"]
            if t_returned is not None:
                info.append(f"returned in {t_returned:.0f} s")
            frame = view.draw(flock, stray, centre, pos, target, phase, barking, info, t - t0)
            if args.show:
                cv2.imshow("Malchy", frame)
                if cv2.waitKey(1) == 27:
                    break
            if t - last_print > 5:
                last_print = t
                print(f"[{t - t0:6.1f}s] {phase:8s} drone=({pos[0]:5.1f},{pos[1]:5.1f},{pos[2]:4.1f}) "
                      f"stray {dist_now:5.1f} m from herd")
            if args.drone == "mavlink":
                time.sleep(max(0.0, 0.1 - (time.time() - wall)))
    finally:
        view.close()
        cv2.destroyAllWindows()

    result = {"stray": name, "start_distance_m": round(start_dist, 1),
              "returned": t_returned is not None,
              "time_to_return_s": round(t_returned, 1) if t_returned is not None else None,
              "herd_moved_m": round(math.dist(start_centre, herd_centre(flock.p, stray)), 1),
              "drone": args.drone, "events": events}
    Path("herding_log.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))
    print("Saved herding_topdown.mp4 and herding_log.json")
    if args.telegram:
        if t_returned is not None:
            notify(f"🐑 Малчы: отбившаяся овца № {name} возвращена к стаду за {t_returned:.0f} с. "
                   f"Дрон вернулся на базу.\n🐑 Малчы: адашкан № {name} кой {t_returned:.0f} секундда "
                   f"оторго кайтарылды.")
        else:
            notify(f"⚠️ Малчы: не удалось вернуть овцу № {name} — проверь её сам.")


if __name__ == "__main__":
    main()
