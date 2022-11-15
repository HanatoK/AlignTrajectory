#!/usr/bin/env python3
import numpy as np


class OptimalRotation:

    def __init__(self, reference_data):
        # reference_data is a N x 3 numpy array
        self.num_atoms = reference_data.shape[0]
        self.reference_positions, self.reference_center = \
            OptimalRotation.bring_to_center(reference_data)
        self.target_positions = None
        self.target_center = None

    @staticmethod
    def bring_to_center(data):
        center_of_geometry = data.mean(axis=0)
        return (data - center_of_geometry), center_of_geometry

    def compute_optimal_rotation_matrix(self, frame_data):
        from numpy.linalg import eigh
        from numpy import array
        self.target_positions, self.target_center = OptimalRotation.bring_to_center(frame_data)
        matrix_F = OptimalRotation.build_matrix_F(pos_target=self.target_positions,
                                                  pos_reference=self.reference_positions,
                                                  num_atoms=self.num_atoms)
        # eigen value decomposition of F
        w, v = eigh(array(matrix_F))
        q = v[:, -1]
        # build rotational matrix
        rotation_matrix = [[0]*3 for _ in range(3)]
        rotation_matrix[0][0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
        rotation_matrix[0][1] = 2.0 * (q[1] * q[2] - q[0] * q[3])
        rotation_matrix[0][2] = 2.0 * (q[1] * q[3] + q[0] * q[2])
        rotation_matrix[1][0] = 2.0 * (q[1] * q[2] + q[0] * q[3])
        rotation_matrix[1][1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
        rotation_matrix[1][2] = 2.0 * (q[2] * q[3] - q[0] * q[1])
        rotation_matrix[2][0] = 2.0 * (q[1] * q[3] - q[0] * q[2])
        rotation_matrix[2][1] = 2.0 * (q[2] * q[3] + q[0] * q[1])
        rotation_matrix[2][2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
        return rotation_matrix, w[-1], q

    def optimal_rmsd(self, frame_data, reference_data=None):
        if reference_data is None:
            reference_data = frame_data
        rotation_matrix, max_eig_val, max_eig_vec = self.compute_optimal_rotation_matrix(frame_data=reference_data)
        s = np.sum(np.square(self.reference_positions) + np.square(self.target_positions))
        return np.sqrt((s - 2.0 * max_eig_val) / self.num_atoms)

    # return two arrays: the first array is the derivative wrt the reference frame, and the second one is the derivative
    # with respect to the target frame
    def optimal_rmsd_derivative(self, frame_data, reference_data=None):
        if reference_data is None:
            reference_data = frame_data
        rotation_matrix, max_eig_val, max_eig_vec = self.compute_optimal_rotation_matrix(frame_data=reference_data)
        s = np.sum(np.square(self.reference_positions) + np.square(self.target_positions))
        # TODO: what should I do if RMSD is zero, 0/0 L'Hôpital's rule??
        factor = 1.0 / (self.num_atoms * np.sqrt((s - 2.0 * max_eig_val) / self.num_atoms))
        diff_ref = self.reference_positions - np.transpose(np.matmul(rotation_matrix, self.target_positions.T))
        diff_tar = self.target_positions - np.transpose(np.matmul(np.transpose(rotation_matrix),
                                                                  self.reference_positions.T))
        return factor * diff_ref, factor * diff_tar

    def optimal_rotate(self, frame_data, reference_data=None):
        if reference_data is None:
            reference_data = frame_data
        rotation_matrix, max_eig_val, max_eig_vec = self.compute_optimal_rotation_matrix(frame_data=reference_data)
        return np.transpose(np.matmul(rotation_matrix, frame_data.T))

    @staticmethod
    def build_matrix_F(pos_target, pos_reference, num_atoms):
        F = [[0]*4 for _ in range(4)]
        R11 = 0
        R22 = 0
        R33 = 0
        R12 = 0
        R13 = 0
        R23 = 0
        R21 = 0
        R31 = 0
        R32 = 0
        for i in range(0, num_atoms):
            R11 += pos_target[i][0] * pos_reference[i][0]
            R22 += pos_target[i][1] * pos_reference[i][1]
            R33 += pos_target[i][2] * pos_reference[i][2]
            R12 += pos_target[i][0] * pos_reference[i][1]
            R13 += pos_target[i][0] * pos_reference[i][2]
            R23 += pos_target[i][1] * pos_reference[i][2]
            R21 += pos_target[i][1] * pos_reference[i][0]
            R31 += pos_target[i][2] * pos_reference[i][0]
            R32 += pos_target[i][2] * pos_reference[i][1]
        F[0][0] = R11 + R22 + R33
        F[0][1] = R23 - R32
        F[0][2] = R31 - R13
        F[0][3] = R12 - R21
        F[1][0] = F[0][1]
        F[1][1] = R11 - R22 - R33
        F[1][2] = R12 + R21
        F[1][3] = R13 + R31
        F[2][0] = F[0][2]
        F[2][1] = F[1][2]
        F[2][2] = -R11 + R22 - R33
        F[2][3] = R23 + R32
        F[3][0] = F[0][3]
        F[3][1] = F[1][3]
        F[3][2] = F[2][3]
        F[3][3] = -R11 - R22 + R33
        return F


def _test_rotation():
    import pandas as pd
    traj = pd.read_csv('../tests/example_rmsd_input.txt', delimiter=r'\s+', comment='#', header=None)
    num_atoms = traj.shape[1] // 3
    # rotated = None
    for i in range(0, traj.shape[0] - 1):
        reference_frame = traj.iloc[i].to_numpy().reshape(-1, 3)
        rot = OptimalRotation(reference_frame)
        for j in range(i + 1, traj.shape[0]):
            target_frame = traj.iloc[j].to_numpy().reshape(-1, 3)
            reference_frame_centered, _ = OptimalRotation.bring_to_center(reference_frame)
            target_frame_centered, _ = OptimalRotation.bring_to_center(target_frame)
            rotated = rot.optimal_rotate(target_frame_centered)
            diff = rotated - reference_frame_centered
            rmsd = np.sqrt(np.sum(np.square(diff)) / num_atoms)
            print(f'Optimal RMSD between frame {i:4d} and {j:4d} is {rmsd:15.12f}')
    # print(rotated)


def _test_optimal_rmsd():
    import pandas as pd
    traj = pd.read_csv('../tests/example_rmsd_input.txt', delimiter=r'\s+', comment='#', header=None)
    num_atoms = traj.shape[1] // 3
    for i in range(0, traj.shape[0] - 1):
        reference_frame = traj.iloc[i].to_numpy().reshape(num_atoms, 3)
        rot = OptimalRotation(reference_frame)
        for j in range(i + 1, traj.shape[0]):
            target_frame = traj.iloc[j].to_numpy().reshape(num_atoms, 3)
            rmsd = rot.optimal_rmsd(target_frame)
            print(f'Optimal RMSD between frame {i:4d} and {j:4d} is {rmsd:15.12f}')


if __name__ == '__main__':
    _test_rotation()
    _test_optimal_rmsd()
