#!/usr/bin/env python3
import time
import numpy as np
import pinocchio as pin

# Your existing modules
from robot_control.robot_arm import (
    G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
)
from robot_control.robot_arm_ik import (
    G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
)


def pick_arm_classes(arm_name):
    """Map robot name to correct controller & IK classes."""
    if arm_name == "G1_29":
        return G1_29_ArmIK, G1_29_ArmController
    if arm_name == "G1_23":
        return G1_23_ArmIK, G1_23_ArmController
    if arm_name == "H1_2":
        return H1_2_ArmIK, H1_2_ArmController
    if arm_name == "H1":
        return H1_ArmIK, H1_ArmController
    raise ValueError(f"Unknown arm type: {arm_name}")


def make_target_se3(xyz, R=None):
    """Convert (x, y, z) and optional rotation to a Pinocchio SE3."""
    if R is None:
        R = np.eye(3)
    return pin.SE3(R, np.array(xyz, dtype=float))


def current_ee_se3s(ik, q_lr):
    """Return left and right end-effector poses from the reduced model."""
    m = ik.reduced_robot.model
    d = ik.reduced_robot.data
    pin.forwardKinematics(m, d, q_lr)
    pin.updateFramePlacements(m, d)
    L = m.getFrameId("L_ee")
    R = m.getFrameId("R_ee")
    return d.oMf[L], d.oMf[R]


def move_arms(
    L_xyz=(0.30, +0.25, 0.20),
    R_xyz=(0.30, -0.25, 0.20),
    hz=60.0,
    motion=True,
    sim=True,
    iface="lo",
    alpha=0.3,
    arm_name="G1_29",
    pos_tolerance=0.01,
    max_seconds=10.0,
):
    """
    Move both arms to specified (x, y, z) positions using inverse kinematics.
    Parameters mimic teleop_hand_and_arm style.
    """
    print(f"Starting move_arms(sim={sim}, iface={iface}, hz={hz})")

    # Orientation (identity or fixed rotation)
    yaw, pitch, roll = 0.0, 0.0, 0.0
    R = pin.rz(yaw) @ pin.ry(pitch) @ pin.rx(roll)

    # Initialize control & IK
    IKCls, CtrlCls = pick_arm_classes(arm_name)
    arm_ik = IKCls(Unit_Test=False, Visualization=False)
    arm_ctl = CtrlCls(simulation_mode=sim, iface=iface)

    if hasattr(arm_ctl, "speed_gradual_max"):
        arm_ctl.speed_gradual_max()

    # Compute goals
    L_goal = make_target_se3(L_xyz, R)
    R_goal = make_target_se3(R_xyz, R)

    # Get current arm state
    q_now = arm_ctl.get_current_dual_arm_q()
    L_now, R_now = current_ee_se3s(arm_ik, q_now)

    def ee_errors(q_lr):
        Lc, Rc = current_ee_se3s(arm_ik, q_lr)
        eL = np.linalg.norm(Lc.translation - L_goal.translation)
        eR = np.linalg.norm(Rc.translation - R_goal.translation)
        return eL, eR

    # Control loop
    rate_dt = 1.0 / hz
    t0 = time.time()

    print(f"Moving to L={L_xyz}, R={R_xyz}")
    while True:
        q = arm_ctl.get_current_dual_arm_q()
        dq = arm_ctl.get_current_dual_arm_dq()

        sol_q, sol_tau = arm_ik.solve_ik(
            L_goal.homogeneous, R_goal.homogeneous, q, dq, alpha=alpha
        )
        arm_ctl.ctrl_dual_arm(sol_q, sol_tau)

        eL, eR = ee_errors(sol_q)
        if eL < pos_tolerance and eR < pos_tolerance:
            print("Target reached.")
            break

        if time.time() - t0 > max_seconds:
            print("Timeout reached.")
            break

        time.sleep(rate_dt)

    if motion:
        print("Holding final pose.")
    else:
        print("Motion disabled; stopping arms.")
        if hasattr(arm_ctl, "ctrl_dual_arm_go_home"):
            arm_ctl.ctrl_dual_arm_go_home()


if __name__ == "__main__":
    # Example usage — just edit the targets here
    move_arms(
        L_xyz=(0.32, 0.26, 0.22),
        R_xyz=(0.32, -0.26, 0.22),
        hz=60.0,
        motion=True,
        sim=True,
        iface="lo",
        alpha=0.3,
    )
