#!/usr/bin/env python3
"""
Minimum-jerk XYZ -> IK demo (dual arms) for Unitree G1_29.

What it does
------------
- Takes XYZ targets for left/right wrists (meters).
- Builds pin.SE3 targets (identity orientation by default).
- Solves IK **once** for the fixed targets to get q_goal.
- Generates a smooth minimum-jerk trajectory from the current q_start
  to q_goal that lasts exactly `--seconds`.
- Publishes at `--hz` using G1_29_ArmController.

Why this avoids snapping
------------------------
We don't stream raw IK every tick. Instead, we time-parameterize a trajectory
q_ref(t) = q_start + s(t) * (q_goal - q_start), where
s(t) = 10u^3 - 15u^4 + 6u^5, u = t/T, T = --seconds.

Usage example
-------------
python3 simple_arm_control_minjerk.py --sim --iface lo \
  --L 0.30 0.25 0.20 --R 0.30 -0.25 0.20 --hz 60 --seconds 6
"""

import argparse
import time
import numpy as np
import pinocchio as pin

# Your SDK imports
from robot_arm import G1_29_ArmController
from robot_arm_ik import G1_29_ArmIK


def se3_from_xyz(xyz, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> pin.SE3:
    """Create pin.SE3 from xyz and (w,x,y,z) quaternion."""
    w, x, y, z = quat_wxyz
    q = pin.Quaternion(w, x, y, z)
    return pin.SE3(q, np.array(xyz, dtype=float))


def s_curve(u: float) -> float:
    """Minimum-jerk S-curve with zero vel/accel at ends; u in [0,1]."""
    u = max(0.0, min(1.0, u))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


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
    # Controller (forward iface if supported; fall back if ctor lacks iface)
    try:
        arm_ctrl = G1_29_ArmController(
            motion_mode=motion, simulation_mode=sim, iface=iface
        )
    except TypeError:
        arm_ctrl = G1_29_ArmController(motion_mode=motion, simulation_mode=sim)

    # IK solver
    arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

    # Targets as SE3 (identity orientation by default)
    L_T = se3_from_xyz(L_xyz)
    R_T = se3_from_xyz(R_xyz)

    # Current state (q_start, dq_start)
    q_start = arm_ctrl.get_current_dual_arm_q()
    dq_start = arm_ctrl.get_current_dual_arm_dq()

    # Solve IK ONCE for the fixed targets to get q_goal
    # Pass 4x4 homogeneous matrices and the warm start
    q_goal, _tau_ff = arm_ik.solve_ik(
        L_T.homogeneous, R_T.homogeneous, q_start, dq_start
    )

    # (optional) set built-in speed limiters if your controller has them
    if hasattr(arm_ctrl, "speed_gradual_max"):
        # Smaller = slower; adjust to taste or comment out if not desired
        arm_ctrl.speed_gradual_max(3.0)

    period = 1.0 / max(1e-3, hz)
    T = max(0.0, float(seconds))
    t0 = time.time()

    # Main loop: follow the time-parameterized profile
    last_log = 0.0
    while True:
        now = time.time()
        u = 1.0 if T <= 0.0 else min(1.0, (now - t0) / T)
        s = s_curve(u)

        # Interpolate joint reference
        q_ref = q_start + s * (q_goal - q_start)

        # Feedforward is optional here; zero is fine for most position controllers
        arm_ctrl.ctrl_dual_arm(q_ref, np.zeros_like(q_ref))

        # Lightweight progress log every ~0.5s
        if now - last_log > 0.5:
            joint_err = np.linalg.norm(q_goal - q_ref)
            print(
                f"[traj] u={u:.2f}, s={s:.2f}, |q_err|={joint_err:.4f} rad"
            )
            last_log = now

        # End when we reach the allotted time (and send the exact final pose once)
        if u >= 1.0:
            arm_ctrl.ctrl_dual_arm(q_goal, np.zeros_like(q_goal))
            print("[traj] Reached target. Holding final pose.")
            break

        # Maintain rate
        loop_dt = time.time() - now
        if loop_dt < period:
            time.sleep(period - loop_dt)

    # Optional: hold final pose a short while for stability/observation
    hold_sec = 0.5
    t_hold_end = time.time() + hold_sec
    while time.time() < t_hold_end:
        arm_ctrl.ctrl_dual_arm(q_goal, np.zeros_like(q_goal))
        time.sleep(period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sim", action="store_true", help="use simulation mode (DDS channel 1)"
    )
    ap.add_argument(
        "--iface",
        default=None,
        help="DDS network interface name (e.g., lo, enp3s0). Forwarded if supported.",
    )
    ap.add_argument(
        "--motion",
        action="store_true",
        help="publish to motion topic (arm_sdk) instead of debug lowcmd",
    )
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument(
        "--L",
        nargs=3,
        type=float,
        default=[0.20, 0.15, 0.05],
        metavar=("X", "Y", "Z"),
        help="Left wrist XYZ (m)",
    )
    ap.add_argument(
        "--R",
        nargs=3,
        type=float,
        default=[0.20, -0.15, 0.05],
        metavar=("X", "Y", "Z"),
        help="Right wrist XYZ (m)",
    )
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
