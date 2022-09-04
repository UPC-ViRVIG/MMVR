import numpy as np
import serializer_helper as sh


class trackers_info_dataset:
    def __init__(self, path):
        self.import_info(path)

    def import_info(self, path):
        # Open as read binary
        with open(path, "rb") as f:
            # Read Header
            self.number_poses = sh.read_uint(f)
            self.number_trackers = sh.read_uint(f)
            self.number_features_tracker = sh.read_uint(f)
            self.number_features = sh.read_uint(f)
            # Mean and Standard Deviation
            self.mean = np.zeros(self.number_features)
            self.std = np.zeros(self.number_features)
            for i in range(self.number_features):
                self.mean[i] = sh.read_float(f)
                self.std[i] = sh.read_float(f)
            # Read Poses
            self.info = np.zeros((self.number_poses, self.number_features))
            for i in range(self.number_poses):
                for j in range(self.number_features):
                    self.info[i][j] = sh.read_float(f)
            # Read Positions
            self.positions = np.zeros((self.number_poses, 3, 3))
            for i in range(self.number_poses):
                for j in range(3):
                    for k in range(3):
                        self.positions[i][j][k] = sh.read_float(f)
