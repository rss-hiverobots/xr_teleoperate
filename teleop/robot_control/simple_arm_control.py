#!/usr/bin/env python3
"""
Minimum-jerk XYZ -> IK demo (dual arms) for Unitree G1_29.

Adds:
- Uses the controller's built-in DDS subscribe wait (constructor).
- Verifies a *stable* starting joint vector before moving.
"""

import argparse
import time
from typing import Tuple
import numpy as np
import pinocchio as pin

# Your SDK imports (module paths as in your files)
from robot_arm import G1_29_ArmController
from robot_arm_ik import G1_29_ArmIK


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def se3_from_xyz(xyz, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> pin.SE3:
    """Create pin.SE3 from xyz and (w,x,y,z) quaternion."""
    w, x, y, z = quat_wxyz
    q = pin.Quaternion(w, x, y, z)
    q.normalize()
    return pin.SE3(q, np.array(xyz, dtype=float))


def s_curve(u: float) -> float:
    """Minimum-jerk S-curve with zero vel/accel at ends; u in [0,1]."""
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def wait_for_stable_arm_q(
    ctrl: G1_29_ArmController,
    window: int = 20,
    std_tol: float = 1e-4,
    max_wait_s: float = 10.0,
    sample_dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probe arm q/dq until samples look stable. Returns (q_start, dq_start).
    """
    buf = []
    t0 = time.time()
    while True:
        q = ctrl.get_current_dual_arm_q()
        dq = ctrl.get_current_dual_arm_dq()
        if q is not None and dq is not None and np.all(np.isfinite(q)) and np.all(np.isfinite(dq)):
            buf.append(q.copy())
            if len(buf) > window:
                buf.pop(0)
                std = np.std(np.stack(buf, axis=0), axis=0)
                if np.max(std) < std_tol:
                    return q, dq
        if time.time() - t0 > max_wait_s:
            # Best-effort fallback
            return q, dq
        time.sleep(sample_dt)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def run_xyz_demo(
    L_xyz=(0.25, +0.25, 0.20),
    R_xyz=(0.25, -0.25, 0.20),
    hz: float = 60.0,
    seconds: float = 6.0,
    motion: bool = True,
    sim: bool = True,
    iface: str | None = None,
):
    """
    Execute a single, smooth, minimum-jerk move from the current arm configuration
    to the IK solution that reaches the given XYZ targets.
    """

    # 1) Controller (constructor blocks until DDS lowstate is live).
    #    Your controller also locks non-arm joints and spawns a publish thread.
    #    iface is supported and forwarded to ChannelFactoryInitialize.
    arm_ctrl = G1_29_ArmController(motion_mode=motion, simulation_mode=sim, iface=iface)

    # Optionally ramp velocity limit gently (uses controller's helper).
    if hasattr(arm_ctrl, "speed_gradual_max"):
        arm_ctrl.speed_gradual_max(5.0)

    # 2) IK solver
    arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

    # 3) Build fixed targets (identity orientation by default)
    L_T = se3_from_xyz(L_xyz)
    R_T = se3_from_xyz(R_xyz)

    # 4) Ensure a stable starting state
    print("[READY] Waiting for stable arm feedback...")
    q_start, dq_start = wait_for_stable_arm_q(arm_ctrl)
    print(f"[READY] q_start shape={q_start.shape}, max|dq|={np.max(np.abs(dq_start)):.4f}")

    # 5) Solve IK ONCE to get q_goal for these fixed targets
    q_goal, _tau_ff = arm_ik.solve_ik(L_T.homogeneous, R_T.homogeneous, q_start, dq_start)

    # 6) Time-parameterized minimum-jerk blend q(t) = q0 + s(t)*(qg - q0)
    period = 1.0 / max(1e-3, hz)
    T = max(0.0, float(seconds))
    t0 = time.time()
    last_log = 0.0

    while True:
        now = time.time()
        u = 1.0 if T <= 0.0 else min(1.0, (now - t0) / T)
        s = s_curve(u)
        q_ref = q_start + s * (q_goal - q_start)

        # Feed zero tau_ff; controller handles rate limiting internally
        arm_ctrl.ctrl_dual_arm(q_ref, np.zeros_like(q_ref))

        if now - last_log > 0.5:
            err = float(np.linalg.norm(q_goal - q_ref))
            print(f"[traj] u={u:.2f}, s={s:.2f}, |q_err|={err:.4f} rad")
            last_log = now

        if u >= 1.0:
            arm_ctrl.ctrl_dual_arm(q_goal, np.zeros_like(q_goal))
            print("[traj] Reached target. Holding final pose.")
            break

        # Maintain user-specified loop rate (publish thread runs at 250 Hz internally)
        loop_dt = time.time() - now
        if loop_dt < period:
            time.sleep(period - loop_dt)

    # 7) Hold briefly for observation
    hold_sec = 0.5
    t_end = time.time() + hold_sec
    while time.time() < t_end:
        arm_ctrl.ctrl_dual_arm(q_goal, np.zeros_like(q_goal))
        time.sleep(period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="use simulation mode (DDS channel 1)")
    ap.add_argument("--iface", default=None, help="DDS network interface (e.g., lo, enp3s0)")
    ap.add_argument("--motion", action="store_true", help="publish to motion topic (arm_sdk)")
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--L", nargs=3, type=float, default=[0.20, 0.15, 0.25], metavar=("X", "Y", "Z"))
    ap.add_argument("--R", nargs=3, type=float, default=[0.20, -0.15, 0.25], metavar=("X", "Y", "Z"))
    args = ap.parse_args()

    try:
        run_xyz_demo(
            L_xyz=tuple(args.L),
            R_xyz=tuple(args.R),
            hz=args.hz,
            seconds=args.seconds,
            motion=args.motion,
            sim=args.sim,
            iface=args.iface,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")


if __name__ == "__main__":
    main()
