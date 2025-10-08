from typing import Tuple

URDF_PATH = "/home/eymen/ros2_ws/src/g1_manipulation/g1_manipulation/g1/g1_body29_hand14.urdf"

# -------------- Joint Names --------------------------

# Joint names as declared in the urdf file
URDF_JOINT_NAMES = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
    "left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint",
    "left_hand_index_0_joint","left_hand_index_1_joint",
    "left_hand_middle_0_joint","left_hand_middle_1_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
    "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
    "right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint",
    "right_hand_index_0_joint","right_hand_index_1_joint",
    "right_hand_middle_0_joint","right_hand_middle_1_joint"
]

# Joint names ordered according to the q vector sent by the robot
JOINT_NAMES = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
    "left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
    "right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"
]

# -------------- Joint Groups --------------------------
JOINT_GROUPS = {
    "left_leg": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint"
    ],
    "right_leg": [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint"
    ],
    "waist": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint"
    ],
    "left_arm": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint"
    ],
    "right_arm": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    ],
    "hands": [
        "left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint",
        "left_hand_index_0_joint","left_hand_index_1_joint",
        "left_hand_middle_0_joint","left_hand_middle_1_joint",
        "right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint",
        "right_hand_index_0_joint","right_hand_index_1_joint",
        "right_hand_middle_0_joint","right_hand_middle_1_joint"
    ]
}

# -------------- Default Locked Joints --------------------------

DEFAULT_LOCKED_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",

    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]

#-------------- Safe Joint Limits (!!Legs not reviewed!!)--------------------------
# Per-joint limits (min, max) in radians; index-aligned with JOINT_NAMES
JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-2.5307,  2.8798),   # 0  L_LEG_HIP_PITCH
    (-0.5236,  2.9671),   # 1  L_LEG_HIP_ROLL
    (-2.7576,  2.7576),   # 2  L_LEG_HIP_YAW
    (-0.087267,2.8798),   # 3  L_LEG_KNEE
    (-0.87267, 0.5236),   # 4  L_LEG_ANKLE_PITCH
    (-0.2618,  0.2618),   # 5  L_LEG_ANKLE_ROLL
    (-2.5307,  2.8798),   # 6  R_LEG_HIP_PITCH
    (-2.9671,  0.5236),   # 7  R_LEG_HIP_ROLL
    (-2.7576,  2.7576),   # 8  R_LEG_HIP_YAW
    (-0.087267,2.8798),   # 9  R_LEG_KNEE
    (-0.87267, 0.5236),   # 10 R_LEG_ANKLE_PITCH
    (-0.2618,  0.2618),   # 11 R_LEG_ANKLE_ROLL
    (-2.618,   2.618),    # 12 WAIST_YAW
    (-0.52,    0.52),     # 13 WAIST_ROLL
    (-0.52,    0.52),     # 14 WAIST_PITCH
    (-2.0,  0.65),   # 15 L_SHOULDER_PITCH
    (-0.1,  2.25),   # 16 L_SHOULDER_ROLL
    (-1.5,   1.4),    # 17 L_SHOULDER_YAW
    (-0.95,  1.65),   # 18 L_ELBOW
    (-1.972222054, 1.972222054),  # 19 L_WRIST_ROLL
    (-1.614429558, 1.614429558),  # 20 L_WRIST_PITCH
    (-1.614429558, 1.614429558),  # 21 L_WRIST_YAW
    (-2.0,  0.65),   # 22 R_SHOULDER_PITCH
    (-2.25,  0.1),   # 23 R_SHOULDER_ROLL
    (-1.5,   1.4),    # 24 R_SHOULDER_YAW
    (-0.95,  1.65),   # 25 R_ELBOW
    (-1.972222054, 1.972222054),  # 26 R_WRIST_ROLL
    (-1.614429558, 1.614429558),  # 27 R_WRIST_PITCH
    (-1.614429558, 1.614429558),  # 28 R_WRIST_YAW
)

""""
#-------------- Real Joint Limits --------------------------
# Per-joint limits (min, max) in radians; index-aligned with JOINT_NAMES
JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-2.5307,  2.8798),   # 0  L_LEG_HIP_PITCH
    (-0.5236,  2.9671),   # 1  L_LEG_HIP_ROLL
    (-2.7576,  2.7576),   # 2  L_LEG_HIP_YAW
    (-0.087267,2.8798),   # 3  L_LEG_KNEE
    (-0.87267, 0.5236),   # 4  L_LEG_ANKLE_PITCH
    (-0.2618,  0.2618),   # 5  L_LEG_ANKLE_ROLL
    (-2.5307,  2.8798),   # 6  R_LEG_HIP_PITCH
    (-2.9671,  0.5236),   # 7  R_LEG_HIP_ROLL
    (-2.7576,  2.7576),   # 8  R_LEG_HIP_YAW
    (-0.087267,2.8798),   # 9  R_LEG_KNEE
    (-0.87267, 0.5236),   # 10 R_LEG_ANKLE_PITCH
    (-0.2618,  0.2618),   # 11 R_LEG_ANKLE_ROLL
    (-2.618,   2.618),    # 12 WAIST_YAW
    (-0.52,    0.52),     # 13 WAIST_ROLL
    (-0.52,    0.52),     # 14 WAIST_PITCH
    (-3.0892,  2.6704),   # 15 L_SHOULDER_PITCH
    (-1.5882,  2.2515),   # 16 L_SHOULDER_ROLL
    (-2.618,   2.618),    # 17 L_SHOULDER_YAW
    (-1.0472,  2.0944),   # 18 L_ELBOW
    (-1.972222054, 1.972222054),  # 19 L_WRIST_ROLL
    (-1.614429558, 1.614429558),  # 20 L_WRIST_PITCH
    (-1.614429558, 1.614429558),  # 21 L_WRIST_YAW
    (-3.0892,  2.6704),   # 22 R_SHOULDER_PITCH
    (-2.2515,  1.5882),   # 23 R_SHOULDER_ROLL
    (-2.618,   2.618),    # 24 R_SHOULDER_YAW
    (-1.0472,  2.0944),   # 25 R_ELBOW
    (-1.972222054, 1.972222054),  # 26 R_WRIST_ROLL
    (-1.614429558, 1.614429558),  # 27 R_WRIST_PITCH
    (-1.614429558, 1.614429558),  # 28 R_WRIST_YAW
)
"""
