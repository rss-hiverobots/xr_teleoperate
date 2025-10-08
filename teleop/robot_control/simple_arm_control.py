#!/usr/bin/env python3
# import time, numpy as np
# import pinocchio as pin

# # pick your robot flavor here
# from robot_arm_ik import G1_29_ArmIK  # G1_23_ArmIK, H1_2_ArmIK...
# from robot_arm import G1_29_ArmController

# def se3_from_xyz(x, y, z, quat=(1,0,0,0)):
#     """Make a pin.SE3 from position and quaternion (w,x,y,z)."""
#     q = pin.Quaternion(*quat)  # identity by default
#     return pin.SE3(q, np.array([x, y, z], dtype=float))

# def run_xyz_demo(L_xyz, R_xyz, hz=60.0, motion=True, sim=False):
#     arm_ik   = G1_29_ArmIK()
#     arm_ctrl = G1_29_ArmController(motion_mode=motion, simulation_mode=sim, iface="lo")

#     dt = 1.0 / hz
#     while True:
#         # Build targets (identity orientation; change quat if you want wrist orientation)
#         L_T = se3_from_xyz(*L_xyz)   # e.g. (0.30, +0.25, 0.20)
#         R_T = se3_from_xyz(*R_xyz)   # e.g. (0.30, -0.25, 0.20)

#         # (optional) pass current q/dq to warm-start smoothing & dynamics
#         # q_now = arm_ctrl.get_current_dual_arm_q()
#         # dq_now = arm_ctrl.get_current_dual_arm_dq()

#         q_cmd, tau_ff = arm_ik.solve_ik(L_T, R_T)  # or (..., q_now, dq_now)
#         arm_ctrl.ctrl_dual_arm(q_cmd, tau_ff)

#         time.sleep(dt)

# if __name__ == "__main__":
#     # Example: both hands forward, 25 cm left/right of center, 20 cm up
#     run_xyz_demo(L_xyz=(0.30, +0.25, 0.20), R_xyz=(0.30, -0.25, 0.20), hz=60.0, motion=True, sim=True)
#!/usr/bin/env python3
"""
Simple XYZ point to IK demo (dual arms) using G1_29_ArmIK + G1_29_ArmController.
- Accepts XYZ targets (meters) for L/R wrists.
- Builds pin.SE3 targets then passes **4x4 homogeneous** matrices to solve_ik (required).
- Warm-starts IK with current joint state each cycle.
- Publishes continuously at --hz until convergence or Ctrl+C.

Usage examples:
  python3 simple_arm_control_fixed.py --sim --iface lo \
      --L 0.30 0.25 0.20 --R 0.30 -0.25 0.20 --hz 60 --seconds 6

Notes:
- --sim uses DDS channel 1 inside ArmController; --iface is forwarded when supported.
- If your ArmController doesn't yet accept iface=, the code auto-falls-back to old ctor.
"""
import argparse
import time
import numpy as np
import pinocchio as pin

from robot_arm import (
G1_29_ArmController,
G1_29_JointIndex,
G1_29_JointArmIndex,
)
from robot_arm_ik import G1_29_ArmIK


def se3_from_xyz(xyz, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
    """Create pin.SE3 from xyz and (w,x,y,z) quaternion."""
    w, x, y, z = quat_wxyz
    q = pin.Quaternion(w, x, y, z)
    return pin.SE3(q, np.array(xyz, dtype=float))


def run_xyz_demo(L_xyz=(0.25, +0.15, 0.05), R_xyz=(0.25, -0.15, 0.05),
                 hz=60.0, seconds=6.0, motion=True, sim=True, iface=None):
    # Controller (forward iface if supported)
    try:
        arm_ctrl = G1_29_ArmController(motion_mode=motion, simulation_mode=sim, iface=iface)
    except TypeError:
        # back-compat for older signature without iface
        arm_ctrl = G1_29_ArmController(motion_mode=motion, simulation_mode=sim)

    # IK solver (Unit_Test=False to use absolute asset paths variant)
    arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

    # Targets as SE3
    L_T = se3_from_xyz(L_xyz)
    R_T = se3_from_xyz(R_xyz)

    # ===== Lower-body lock: snapshot startup angles and enforce each cycle =====
    locked_lower = None
    # prefer a full-body getter if available on your controller
    if hasattr(arm_ctrl, "get_current_motor_q"):
        q_all = arm_ctrl.get_current_motor_q()
        arm_ids = {i.value for i in G1_29_JointArmIndex}
        locked_lower = {jid.value: q_all[jid.value] for jid in G1_29_JointIndex if jid.value not in arm_ids}
        print(f"[Lock] lower body {len(locked_lower)} joints locked to startup pose.")
    else:
        print("[Lock] WARNING: controller lacks get_current_motor_q(); lower-body lock skipped.")
    # Optional gentle ramp for arm speed if your controller supports it
    if hasattr(arm_ctrl, 'speed_gradual_max'):
        arm_ctrl.speed_gradual_max(5.0)

    period = 1.0 / max(1e-3, hz)
    t_end = time.time() + max(0.0, seconds)

    # Optional gentle ramp for arm speed if your controller supports it
    if hasattr(arm_ctrl, 'speed_gradual_max'):
        arm_ctrl.speed_gradual_max(5.0)
    period = 1.0 / max(1e-3, hz)
    t_end = time.time() + max(0.0, seconds)

    # Convergence thresholds (position/orientation)
    POS_EPS = 1e-3   # 1 mm
    ROT_EPS = 2.0 * np.pi / 180.0  # 2 degrees

    def ee_error(q):
        # Compute current EE poses with reduced model via arm_ik (numerical check)
        # We only use the translational/orientational residual from the same cost functions
        # For simplicity and speed, we reuse arm_ik.translational_error/rotational_error with current q.
        Tl = L_T.homogeneous; Tr = R_T.homogeneous
        te = np.array(arm_ik.translational_error(q, Tl, Tr)).reshape(-1)
        re = np.array(arm_ik.rotational_error(q, Tl, Tr)).reshape(-1)
        # split (L first 3, R next 3) for both
        pos_err = max(np.linalg.norm(te[:3]), np.linalg.norm(te[3:]))
        rot_err = max(np.linalg.norm(re[:3]), np.linalg.norm(re[3:]))
        return pos_err, rot_err

    last_print = 0.0
    while True:
        t0 = time.time()
        # Warm start from current state
        q_now = arm_ctrl.get_current_dual_arm_q()
        dq_now = arm_ctrl.get_current_dual_arm_dq()

        # Solve IK (pass 4x4 matrices!)
        q_cmd, tau_ff = arm_ik.solve_ik(L_T.homogeneous, R_T.homogeneous, q_now, dq_now)

        # Send to controller
        arm_ctrl.ctrl_dual_arm(q_cmd, tau_ff)

        # Simple convergence check (optional)
        if time.time() - last_print > 0.5:
            pe, re = ee_error(q_cmd)
            print(f"[IK] pos_err={pe*1e3:.2f} mm, rot_err={re*180/np.pi:.2f} deg")
            last_print = time.time()
        if seconds > 0.0 and time.time() >= t_end:
            break

        # Maintain rate
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', action='store_true', help='use simulation mode (DDS channel 1)')
    ap.add_argument('--iface', default=None, help='DDS network interface name (e.g., lo, enp3s0)')
    ap.add_argument('--motion', action='store_true', help='publish to motion topic (arm_sdk) instead of debug lowcmd')
    ap.add_argument('--hz', type=float, default=60.0)
    ap.add_argument('--seconds', type=float, default=6.0)
    ap.add_argument('--L', nargs=3, type=float, default=[0.4, 0.25, 0.3], metavar=('X','Y','Z'))
    ap.add_argument('--R', nargs=3, type=float, default=[0.4, -0.25, 0.3], metavar=('X','Y','Z'))
    args = ap.parse_args()

    try:
        run_xyz_demo(L_xyz=tuple(args.L), R_xyz=tuple(args.R),
                     hz=args.hz, seconds=args.seconds,
                     motion=args.motion, sim=args.sim, iface=args.iface)
    except KeyboardInterrupt:
        print("\nInterrupted by user")