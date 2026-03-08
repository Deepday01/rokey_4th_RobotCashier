from .packing_models import Pose3D


def normalize_pose_rpy(pose: Pose3D, logger=None) -> Pose3D:
    return Pose3D(
        x=pose.x,
        y=pose.y,
        z=pose.z,
        roll=pose.roll,
        pitch=pose.pitch,
        yaw=pose.yaw,
    )
