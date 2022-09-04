import numpy as np
import serializer_helper as sh


class pose_dataset:
    def __init__(self, path, only_mean_std=False):
        if only_mean_std:
            self.import_mean_std(path)
        else:
            self.import_poses(path)

    def import_poses(self, path):
        # Open as read binary
        with open(path, "rb") as f:
            # Read Header
            self.number_poses = sh.read_uint(f)
            self.number_features_pose = sh.read_uint(f)
            self.number_features_hips = sh.read_uint(f)
            self.number_joints = sh.read_uint(f)
            # Mean and Standard Deviation
            number_features = self.number_features_pose + self.number_features_hips
            self.mean = np.zeros(number_features, dtype=np.float32)
            self.std = np.zeros(number_features, dtype=np.float32)
            for i in range(number_features):
                self.mean[i] = sh.read_float(f)
                self.std[i] = sh.read_float(f)
            # JointLocalOffsets
            self.joint_local_offsets = np.zeros(
                (self.number_joints, 3), dtype=np.float32
            )
            for i in range(self.number_joints):
                self.joint_local_offsets[i][0] = sh.read_float(f)
                self.joint_local_offsets[i][1] = sh.read_float(f)
                self.joint_local_offsets[i][2] = sh.read_float(f)
            # Read Poses
            self.poses = np.zeros((self.number_poses, self.number_features_pose))
            for i in range(self.number_poses):
                for j in range(self.number_features_pose):
                    self.poses[i][j] = sh.read_float(f)
            # Read Hips
            self.hips = np.zeros((self.number_poses, self.number_features_hips))
            for i in range(self.number_poses):
                for j in range(self.number_features_hips):
                    self.hips[i][j] = sh.read_float(f)

    def import_mean_std(self, path):
        # Open as read binary
        with open(path, "rb") as f:
            # Read Header
            self.number_poses = sh.read_uint(f)
            self.number_features_pose = sh.read_uint(f)
            self.number_features_hips = sh.read_uint(f)
            self.number_joints = sh.read_uint(f)
            # Mean and Standard Deviation
            number_features = self.number_features_pose + self.number_features_hips
            self.mean = np.zeros(number_features, dtype=np.float32)
            self.std = np.zeros(number_features, dtype=np.float32)
            for i in range(number_features):
                self.mean[i] = sh.read_float(f)
                self.std[i] = sh.read_float(f)
