#!/usr/bin/env python3
"""
go_to_joint_target.py — DDS-once refactor (G1 + MuJoCo/Robot auto-switch)

- iface == "lo"  -> MuJoCo (domain=1, pub 'rt/lowcmd', explicit pos mode, no enable flag)
- iface != "lo"  -> Robot  (domain=0, pub 'rt/arm_sdk', vendor enable flag on joint 29)
"""

import time
import threading
from collections import deque
from typing import Sequence, Optional
import sys
import numpy as np

from params import JOINT_NAMES, JOINT_LIMITS

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.utils.crc import CRC

# -------------------------- Constants --------------------------
N = 29
HZ = 500.0
DT = 1.0 / HZ

# PD gains
KP = np.array([
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40
], dtype=float)

KD = np.array([
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1
], dtype=float)

# -------------------------- G1 Joint Map -----------------------
class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4  # AnkleB
    LeftAnkleRoll = 5   # AnkleA
    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10  # AnkleB
    RightAnkleRoll = 11   # AnkleA
    # Waist
    WaistYaw = 12
    WaistRoll = 13        # may be invalid when locked
    WaistPitch = 14       # may be invalid when locked
    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # invalid for 23DoF
    LeftWristYaw = 21     # invalid for 23DoF
    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # invalid for 23DoF
    RightWristYaw = 28    # invalid for 23DoF
    # Vendor "weight"/enable channel used by the example to toggle arm SDK
    kNotUsedJoint = 29

ARM_JOINTS = [
    G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,      G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
    G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
    G1JointIndex.WaistYaw,
    G1JointIndex.WaistRoll,
    G1JointIndex.WaistPitch,
]

# -------------------------- Context chooser --------------------
def _context_from_iface(iface: str):
    """
    Decide domain/topic/flags from iface:
      - 'lo'   -> MuJoCo bridge (domain 1, lowcmd topic, explicit position mode, ignore enable flag)
      - other  -> Robot (domain 0, arm_sdk topic, stream enable flag)
    """
    if iface == "lo":
        return {
            "domain": 1,
            "nic": "lo",
            "pub_topic": "rt/lowcmd",
            "assert_enable": False,
            "force_pos_mode": True,  # many sims like an explicit position mode
        }
    # default NIC on domain 0 if iface is "auto"/""/None
    if iface in ("", "auto", None):
        nic = None
    else:
        nic = iface
    return {
        "domain": 0,
        "nic": nic,
        "pub_topic": "rt/arm_sdk",
        "assert_enable": True,
        "force_pos_mode": False,
    }

# ==============================================================
# DDS manager (singleton)
# ==============================================================
class _DDS:
    """Owns one pub and one sub and a small message queue."""
    def __init__(self):
        self._lock = threading.RLock()
        self._iface: Optional[str] = None
        self._pub: Optional[ChannelPublisher] = None
        self._sub: Optional[ChannelSubscriber] = None
        self._crc = CRC()
        self._q = deque(maxlen=5)
        self._sub_inited = False
        self._pub_topic = "rt/arm_sdk"

    def init(self, iface: str = "enx98fc84ec937b"):
        ctx = _context_from_iface(iface)
        with self._lock:
            if self._iface == iface and self._pub is not None and self._sub_inited:
                return

            # Domain + NIC
            if ctx["nic"] is None:
                ChannelFactoryInitialize(ctx["domain"])
            else:
                ChannelFactoryInitialize(ctx["domain"], ctx["nic"])

            # Publisher on the chosen topic
            self._pub_topic = ctx["pub_topic"]
            self._pub = ChannelPublisher(self._pub_topic, LowCmd_)
            self._pub.Init()

            # Subscriber (same for sim/robot)
            self._sub = ChannelSubscriber("rt/lowstate", LowState_)
            def _cb(msg):
                self._q.append(msg)
            self._sub.Init(_cb, 10)

            self._sub_inited = True
            self._iface = iface

    def publisher(self) -> ChannelPublisher:
        with self._lock:
            if self._pub is None:
                raise RuntimeError("DDS not initialized. Call dds.init() first.")
            return self._pub

    def pub_topic(self) -> str:
        return self._pub_topic

    def crc(self) -> CRC:
        return self._crc

    def read_lowstate(self, timeout: float = 3.0) -> LowState_:
        t0 = time.monotonic()
        if self._q:
            return self._q.pop()
        while (time.monotonic() - t0) < timeout:
            if self._q:
                return self._q.pop()
            time.sleep(0.002)
        raise TimeoutError("No LowState received (is the bridge running?)")

# global singleton
dds = _DDS()

# ==============================================================
# Helpers
# ==============================================================
def _quintic_scale(t: float, T: float):
    if T <= 0.0:
        return 1.0, 0.0
    tau = max(0.0, min(1.0, t / T))
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    sdot = (30*tau**2 - 60*tau**3 + 30*tau**4) / max(T, 1e-6)
    return s, sdot

def _sync_duration(q0: np.ndarray, qf: np.ndarray, mask: np.ndarray, vmax: float, amax: float) -> float:
    eps = 1e-6
    dq = np.abs(qf - q0) * mask
    T_v = np.where(dq > eps, 1.875 * dq / max(eps, vmax), 0.0)
    T_a = np.where(dq > eps, np.sqrt(11.25 * dq / max(eps, amax)), 0.0)
    T = float(np.max(np.maximum(T_v, T_a)))
    return max(T, 0.5)

def _apply_joint_limits(targets: np.ndarray, strict: bool) -> np.ndarray:
    clamped = targets.copy()
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        v = clamped[i]
        if v < lo or v > hi:
            if strict:
                name = JOINT_NAMES[i]
                raise ValueError(f"Target for {i}:{name}={v:.6f} outside limits [{lo:.6f}, {hi:.6f}]")
            clipped = min(max(v, lo), hi)
            if clipped != v:
                name = JOINT_NAMES[i]
                print(f"[WARN] Clamping {i}:{name} from {v:.6f} to {clipped:.6f} (limits [{lo:.6f}, {hi:.6f}])")
                clamped[i] = clipped
    return clamped

_DEF_ENABLE_VAL = 1.0  # 1: Enable arm_sdk, 0: Disable arm_sdk

# ==============================================================
# Public API
# ==============================================================
def move_g1_to_joint_target(
    targets: Sequence[Optional[float]],
    *,
    iface: str = "enx98fc84ec937b",
    total_time: Optional[float] = None,
    vmax: float = 0.5,
    amax: float = 1.0,
    hold_s: float = 1.0,
    deg: bool = False,
    strict_limits: bool = False,
    arm_only: bool = False,
    release_after: bool = True,
    release_time: float = 1.0,
) -> None:
    """
    Smoothly move joints from CURRENT positions to the provided targets.

    - If `arm_only=True`, only arm/waist joints are commanded; others are held at current.
    - MuJoCo vs Robot behavior auto-selected from `iface`.
    """
    if len(targets) != N:
        raise ValueError(f"Expected {N} targets, got {len(targets)}")

    # Ensure shared DDS is up
    dds.init(iface)
    ctx = _context_from_iface(iface)

    # Fetch current state via shared subscriber
    state = dds.read_lowstate(timeout=5.0)
    motor_arr = getattr(state, "motor_state", None) or getattr(state, "motorState")
    q0 = np.array([float(motor_arr[i].q) for i in range(N)], dtype=float)

    # Masking
    if arm_only:
        mask = np.zeros(N, dtype=float)
        mask[ARM_JOINTS] = 1.0
    else:
        mask = np.array([0 if t is None else 1 for t in targets], dtype=float)

    # Build final targets, with degrees option
    qf_raw = np.array([
        float(q0[i]) if (targets[i] is None or (arm_only and i not in ARM_JOINTS))
        else float(np.deg2rad(targets[i]) if deg else targets[i])
        for i in range(N)
    ], dtype=float)
    qf = _apply_joint_limits(qf_raw, strict=strict_limits)

    T = max(0.2, float(total_time)) if (total_time is not None) else _sync_duration(q0, qf, mask, vmax=vmax, amax=amax)
    print(f"[INFO] pub='{dds.pub_topic()}', domain={ctx['domain']}, T={T:.2f}s, arm_only={arm_only}")

    pub = dds.publisher()
    crc = dds.crc()

    # Prepare cmd (with gains once)
    cmd = LowCmdDefault()
    for i in range(N):
        if ctx["force_pos_mode"]:
            cmd.motor_cmd[i].mode = 0x0A  # explicit position mode (sim)
        cmd.motor_cmd[i].kp = float(KP[i])
        cmd.motor_cmd[i].kd = float(KD[i])
        cmd.motor_cmd[i].tau = 0.0

    # Stream vendor enable flag only on real robot
    if ctx["assert_enable"]:
        cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = _DEF_ENABLE_VAL

    # Motion loop
    t0 = time.perf_counter()
    done = False
    while not done:
        now = time.perf_counter()
        t = now - t0
        if t >= T:
            s, sdot = 1.0, 0.0
            done = True
        else:
            s, sdot = _quintic_scale(t, T)

        qd = q0 + mask * (s * (qf - q0))
        dqd = mask * (sdot * (qf - q0))

        for i in range(N):
            cmd.motor_cmd[i].q = float(qd[i])
            cmd.motor_cmd[i].dq = float(dqd[i])
        if ctx["assert_enable"]:
            cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = _DEF_ENABLE_VAL

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        sleep_t = DT - (time.perf_counter() - now)
        if sleep_t > 0:
            time.sleep(sleep_t)

    # Hold final posture (and keep enable flag if on robot)
    end_time = time.perf_counter() + max(0.0, hold_s)
    while time.perf_counter() < end_time:
        for i in range(N):
            cmd.motor_cmd[i].q = float(qf[i])
            cmd.motor_cmd[i].dq = 0.0
        if ctx["assert_enable"]:
            cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = _DEF_ENABLE_VAL
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        time.sleep(DT)

    # Optional enable fade on robot (kept disabled by default for safety)
    # if ctx["assert_enable"] and release_after and release_time > 0.0:
    #     t0 = time.perf_counter()
    #     while True:
    #         t = time.perf_counter() - t0
    #         val = 0.0 if t >= release_time else float(1.0 - t / release_time)
    #         for i in range(N):
    #             cmd.motor_cmd[i].q = float(qf[i])
    #             cmd.motor_cmd[i].dq = 0.0
    #         cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = val
    #         cmd.crc = crc.Crc(cmd)
    #         pub.Write(cmd)
    #         if t >= release_time:
    #             break
    #         time.sleep(DT)


class JointPostureHold:
    """Background helper that streams a fixed joint posture until stop()."""
    def __init__(self, targets: Sequence[float], *, iface: str = "enx98fc84ec937b", assert_enable: bool = True) -> None:
        if len(targets) != N:
            raise ValueError(f"Expected {N} targets, got {len(targets)}")

        self._targets = np.asarray(targets, dtype=float)
        self._stop_evt = threading.Event()
        self._stopped = False
        self._assert_enable = assert_enable

        dds.init(iface)
        self._ctx = _context_from_iface(iface)
        self._pub = dds.publisher()
        self._crc = dds.crc()

        self._cmd = LowCmdDefault()
        for i in range(N):
            if self._ctx["force_pos_mode"]:
                self._cmd.motor_cmd[i].mode = 0x0A
            self._cmd.motor_cmd[i].kp = float(KP[i])
            self._cmd.motor_cmd[i].kd = float(KD[i])
            self._cmd.motor_cmd[i].tau = 0.0

        self._thread = threading.Thread(target=self._run, name="joint_posture_hold", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            for i in range(N):
                self._cmd.motor_cmd[i].q = float(self._targets[i])
                self._cmd.motor_cmd[i].dq = 0.0
            if self._assert_enable and self._ctx["assert_enable"]:
                self._cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = _DEF_ENABLE_VAL
            self._cmd.crc = self._crc.Crc(self._cmd)
            self._pub.Write(self._cmd)
            time.sleep(DT)

    def stop(self, *, release_time: float = 0.5) -> None:
        if self._stopped:
            return
        self._stop_evt.set()
        self._thread.join(timeout=1.0)
        # Optional small fade of enable flag on robot only
        if self._ctx["assert_enable"] and release_time > 0.0:
            t0 = time.perf_counter()
            while True:
                t = time.perf_counter() - t0
                val = 0.0 if t >= release_time else float(1.0 - t / release_time)
                for i in range(N):
                    self._cmd.motor_cmd[i].dq = 0.0
                self._cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = val
                self._cmd.crc = self._crc.Crc(self._cmd)
                self._pub.Write(self._cmd)
                if t >= release_time:
                    break
                time.sleep(DT)
        self._stopped = True


def get_joint_positions(*, iface: str = "enx98fc84ec937b", timeout: float = 5.0) -> np.ndarray:
    """Return current 29 joint positions using the shared subscriber (no re-init)."""
    dds.init(iface)
    state = dds.read_lowstate(timeout=timeout)
    motor_arr = getattr(state, "motor_state", None) or getattr(state, "motorState")
    return np.array([float(motor_arr[i].q) for i in range(N)], dtype=float)

# -------------------------- Example CLI --------------------------
if __name__ == "__main__":
    #iface = sys.argv[1] if len(sys.argv) > 1 else "enx98fc84ec937b"
    iface = "lo"
    ctx = _context_from_iface(iface)
    print(f"[INFO] Using iface='{iface}' -> pub='{ctx['pub_topic']}', domain={ctx['domain']}")

    print("WARNING: Ensure the robot/sim is clear before commanding motion.")
    # Example: hold first 15 at current, set the 14 arm/hand joints to 0.0 rad, command arm joints only
    targets = [None]*15 + [0.0]*14
    move_g1_to_joint_target(
        targets,
        iface=iface,
        vmax=0.5,
        amax=1.0,
        hold_s=3.0,
        strict_limits=False,
        arm_only=True,
        release_after=True,
        release_time=1.0,
    )
