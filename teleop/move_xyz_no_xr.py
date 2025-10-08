#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal dual-arm IK drive for Unitree G1_29 with debug logging and safety guards.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from robot_control.robot_arm_ik import G1_29_ArmIK
from robot_control.robot_arm import G1_29_ArmController


# ==============================================================================
# Configuration
# ==============================================================================

DT: float = 0.01                     # control period (s)
MAX_JOINT_VEL: float = 0.3           # rad/s
MAX_JOINT_ACC: float = 0.7           # rad/s^2
SMOOTHING_ALPHA: float = 0.2         # 0..1 blend toward IK
POS_TOL_NORM_RAD: float = np.deg2rad(1.0)

# Debugging
DEBUG: bool = True
LOG_EVERY: int = 20                  # log every N control ticks
PRINT_POSES: bool = True

# Neutral orientation for the wrist (xyzw)
NEUTRAL_Q: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

# Base → torso_link (R = I, so only a translation)
T_BASE_TORSO: np.ndarray = np.array([-0.0039635, 0.0, 0.044])

TARGETS: Dict[str, Dict[str, Optional[np.ndarray]]] = {
    "right": {"xyz": np.array([0.20, -0.15, 0.05]), "quat": None},
    "left":  {"xyz": np.array([0.20,  0.15, 0.05]), "quat": None},
}


# ==============================================================================
# Utility: formatting / guards
# ==============================================================================

def _is_finite(arr: np.ndarray) -> bool:
    return np.all(np.isfinite(arr))

def _fmt_vec(v: np.ndarray, prec: int = 3, maxn: int = 6) -> str:
    v = np.asarray(v).flatten()
    head = ", ".join(f"{x:.{prec}f}" for x in v[:maxn])
    if v.size > maxn:
        head += ", ..."
    return f"[{head}] (n={v.size})"

def _fmt_pose(T: np.ndarray) -> str:
    t = T[:3, 3]
    R = T[:3, :3]
    return f"t={_fmt_vec(t)}, R00..02={_fmt_vec(R[0, :], maxn=3)}"

def _safe_vec(name: str, v: np.ndarray) -> np.ndarray:
    if not _is_finite(v):
        print(f"[WARN] {name} had non-finite values; replacing with zeros.")
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v


# ==============================================================================
# Pose utilities
# ==============================================================================

def torso_to_base(xyz_torso: np.ndarray) -> np.ndarray:
    """Convert torso-frame position to base-frame position (R = I)."""
    return np.asarray(xyz_torso, dtype=float) + T_BASE_TORSO

def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    x, y, z, w = q / np.linalg.norm(q)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
            [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )

def se3_from_xyz_quat_base(xyz_base: np.ndarray, quat_xyzw: Optional[np.ndarray]) -> np.ndarray:
    """Build a 4x4 SE(3) pose in the BASE frame from position and quaternion (xyzw)."""
    q = NEUTRAL_Q if quat_xyzw is None else quat_xyzw
    R = quat_xyzw_to_R(q)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(xyz_base, dtype=float)
    return T


# ==============================================================================
# Rate limiting
# ==============================================================================

def rate_limit_step(
    q_cmd_prev: np.ndarray,
    v_prev: np.ndarray,
    q_goal: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One step of velocity/acceleration-constrained motion toward q_goal.
    Returns (q_next, v_cap, dq_step).
    """
    dq = q_goal - q_cmd_prev
    sign = np.sign(dq)

    v_des = v_prev + sign * MAX_JOINT_ACC * dt
    v_cap = np.clip(v_des, -MAX_JOINT_VEL, MAX_JOINT_VEL)

    dq_step = np.clip(dq, -np.abs(v_cap) * dt, np.abs(v_cap) * dt)
    q_next = q_cmd_prev + dq_step

    return q_next, v_cap, dq_step


# ==============================================================================
# Main motion routine
# ==============================================================================

def move_both_arms(ik: G1_29_ArmIK, ctrl: G1_29_ArmController) -> None:
    """Build left/right wrist poses, solve dual-arm IK, smooth & rate-limit, then stream."""
    if DEBUG:
        print("[DBG] Building target wrist poses...")

    # Build BOTH wrist poses as 4x4 in BASE frame
    lp_xyz_base = torso_to_base(TARGETS["left"]["xyz"])
    rp_xyz_base = torso_to_base(TARGETS["right"]["xyz"])

    left_pose  = se3_from_xyz_quat_base(lp_xyz_base,  TARGETS["left"]["quat"])
    right_pose = se3_from_xyz_quat_base(rp_xyz_base, TARGETS["right"]["quat"])

    if PRINT_POSES:
        print(f"[DBG] Left  target (BASE): xyz={_fmt_vec(lp_xyz_base)} | { _fmt_pose(left_pose) }")
        print(f"[DBG] Right target (BASE): xyz={_fmt_vec(rp_xyz_base)} | { _fmt_pose(right_pose) }")

    # Current joint state (14 = 7L + 7R)
    q_curr = _safe_vec("q_curr", ctrl.get_current_dual_arm_q())     # (14,)
    dq_curr = _safe_vec("dq_curr", ctrl.get_current_dual_arm_dq())  # (14,)

    if DEBUG:
        print(f"[DBG] q_curr: min={q_curr.min():.3f}, max={q_curr.max():.3f}, ‖q‖={np.linalg.norm(q_curr):.3f}")
        print(f"[DBG] dq_curr: max|.|={np.max(np.abs(dq_curr)):.3f}, ‖dq‖={np.linalg.norm(dq_curr):.3f}")

    # Optional: joint limits if available
    q_min = q_max = None
    if hasattr(ctrl, "get_joint_limits"):
        try:
            q_min, q_max = ctrl.get_joint_limits()  # expected shape (14,), (14,)
            if DEBUG:
                print(f"[DBG] Limits available. min={_fmt_vec(q_min, prec=2)} | max={_fmt_vec(q_max, prec=2)}")
        except Exception as e:
            if DEBUG:
                print(f"[DBG] Could not read joint limits: {e!r}")

    # IK
    if DEBUG:
        print("[DBG] Solving dual-arm IK...")
    try:
        sol_q, sol_tau_ff = ik.solve_ik(left_pose, right_pose, q_curr, dq_curr)
    except Exception as e:
        print(f"[ERR] IK solver threw: {e!r}")
        return

    sol_q = _safe_vec("sol_q", sol_q)

    if q_min is not None and q_max is not None:
        # warn if solution violates limits
        viol = (sol_q < q_min - 1e-6) | (sol_q > q_max + 1e-6)
        if np.any(viol):
            idx = np.where(viol)[0]
            print(f"[WARN] IK solution outside limits at joints: {idx.tolist()}")
            if DEBUG:
                print(f"[DBG] Offending values: {_fmt_vec(sol_q[idx], prec=3)}")

    # Optional smoothing toward target to avoid big jumps
    q_tgt = SMOOTHING_ALPHA * sol_q + (1.0 - SMOOTHING_ALPHA) * q_curr
    q_tgt = _safe_vec("q_tgt", q_tgt)

    if DEBUG:
        dq_goal = q_tgt - q_curr
        print(f"[DBG] |q_tgt - q_curr|∞ = {np.max(np.abs(dq_goal)):.3f} rad, ‖·‖2 = {np.linalg.norm(dq_goal):.3f}")
        print(f"[DBG] Left Δ (0:7)  max|.|={np.max(np.abs(dq_goal[:7])):.3f}")
        print(f"[DBG] Right Δ (7:14) max|.|={np.max(np.abs(dq_goal[7:])):.3f}")

    # Rate-limited streaming on the full vector
    q_cmd = q_curr.copy()
    v_prev = np.zeros_like(q_cmd)

    err_prev = np.linalg.norm(q_tgt - q_cmd)
    worse_count = 0
    step = 0

    if DEBUG:
        print("[DBG] Entering control loop...")

    while True:
        err = np.linalg.norm(q_tgt - q_cmd)
        if err <= POS_TOL_NORM_RAD:
            if DEBUG:
                print(f"[DBG] Converged: err={err:.4f} rad ≤ tol={POS_TOL_NORM_RAD:.4f}. Exiting.")
            break

        q_cmd_next, v_cap, dq_step = rate_limit_step(q_cmd, v_prev, q_tgt, DT)

        # Saturation diagnostics
        cap_mask = (np.abs(v_cap) >= (MAX_JOINT_VEL - 1e-6))
        if np.any(cap_mask) and (step % LOG_EVERY == 0):
            idx = np.where(cap_mask)[0].tolist()
            print(f"[WARN] Velocity cap active @ joints {idx}; max|v|={np.max(np.abs(v_cap)):.3f} rad/s")

        if not _is_finite(q_cmd_next):
            print("[ERR] Non-finite q_cmd_next detected. Aborting control loop.")
            break

        # (Optional) clamp to limits just-in-case
        if (q_min is not None) and (q_max is not None):
            q_cmd_next = np.clip(q_cmd_next, q_min, q_max)

        # Send command
        try:
            ctrl.ctrl_dual_arm(q_cmd_next, np.zeros_like(q_cmd_next))
        except Exception as e:
            print(f"[ERR] ctrl_dual_arm() threw: {e!r}. Aborting.")
            break

        # Periodic logging
        if DEBUG and (step % LOG_EVERY == 0):
            print(
                f"[DBG] step={step:05d} | err={err:.4f} | "
                f"max|dq_step|={np.max(np.abs(dq_step)):.4f} | "
                f"max|v_cap|={np.max(np.abs(v_cap)):.4f} | "
                f"q_cmd[0:4]={_fmt_vec(q_cmd_next[:4], prec=3, maxn=4)}"
            )

            # Detect divergence (error increasing across logs)
            if err > err_prev + 1e-6:
                worse_count += 1
                print(f"[WARN] Error increased ({worse_count}/3): prev={err_prev:.4f} -> now={err:.4f}")
                if worse_count >= 3:
                    print("[ERR] Error increasing repeatedly—possible instability. Stopping.")
                    break
            else:
                worse_count = 0
            err_prev = err

        # Advance state
        q_cmd, v_prev = q_cmd_next, v_cap
        step += 1
        time.sleep(DT)


# ==============================================================================
# Entrypoint
# ==============================================================================

if __name__ == "__main__":
    print("[INFO] Starting Unitree G1_29 IK test (no XR)...")
    print(f"[INFO] DT={DT}s, v_max={MAX_JOINT_VEL} rad/s, a_max={MAX_JOINT_ACC} rad/s^2, "
          f"smooth={SMOOTHING_ALPHA}, tol={POS_TOL_NORM_RAD:.4f} rad")
    print(f"[INFO] Targets (torso): L={_fmt_vec(TARGETS['left']['xyz'])} | "
          f"R={_fmt_vec(TARGETS['right']['xyz'])}; base←torso t={_fmt_vec(T_BASE_TORSO)}")

    ik = G1_29_ArmIK()
    ctrl = G1_29_ArmController(simulation_mode=True, iface="lo")

    move_both_arms(ik, ctrl)

    print("[INFO] Done.")
