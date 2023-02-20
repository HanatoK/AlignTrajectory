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
                                                  pos_reference=self.reference_positions)
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
    def build_matrix_F(pos_target, pos_reference):
        mat_R = np.matmul(pos_target.T, pos_reference)
        F00 = mat_R[0][0] + mat_R[1][1] + mat_R[2][2]
        F01 = mat_R[1][2] - mat_R[2][1]
        F02 = mat_R[2][0] - mat_R[0][2]
        F03 = mat_R[0][1] - mat_R[1][0]
        F10 = F01
        F11 = mat_R[0][0] - mat_R[1][1] - mat_R[2][2]
        F12 = mat_R[0][1] + mat_R[1][0]
        F13 = mat_R[0][2] + mat_R[2][0]
        F20 = F02
        F21 = F12
        F22 = -mat_R[0][0] + mat_R[1][1] - mat_R[2][2]
        F23 = mat_R[1][2] + mat_R[2][1]
        F30 = F03
        F31 = F13
        F32 = F23
        F33 = -mat_R[0][0] - mat_R[1][1] + mat_R[2][2]
        F = np.asarray([[F00, F01, F02, F03],
                        [F10, F11, F12, F13],
                        [F20, F21, F22, F23],
                        [F30, F31, F32, F33]])
        return F


if __name__ == '__main__':
    def _test_rotation():
        import pandas as pd
        reference = pd.read_csv('tests/reference_frame.txt', delimiter=r'\s+', comment='#', header=None)
        traj = pd.read_csv('tests/example_rmsd_input.txt', delimiter=r'\s+', comment='#', header=None)
        num_atoms = traj.shape[1] // 3
        reference_frame = reference.iloc[0].to_numpy().reshape(-1, 3)
        rot = OptimalRotation(reference_frame)
        reference_frame_centered, _ = OptimalRotation.bring_to_center(reference_frame)
        rmsds = []
        for i in range(0, traj.shape[0]):
            target_frame = traj.iloc[i].to_numpy().reshape(-1, 3)
            target_frame_centered, _ = OptimalRotation.bring_to_center(target_frame)
            rotated = rot.optimal_rotate(target_frame_centered)
            diff = rotated - reference_frame_centered
            rmsd = np.sqrt(np.sum(np.square(diff)) / num_atoms)
            rmsds.append(rmsd)
            print(f'Optimal RMSD between frame {i:4d} and reference is {rmsd:15.12f}')
        # compare the results with those from Colvars
        rmsds_colvars = pd.read_csv(
            'tests/example_rmsd_colvars.txt',
            delimiter=r'\s+', comment='#', header=None).iloc[:, 0].to_numpy()
        rmsds = np.asarray(rmsds)
        error = np.sqrt(np.mean(np.square(rmsds - rmsds_colvars)))
        print(f'Error = {error}')
        # print(rotated)


    def _test_optimal_rmsd():
        import pandas as pd
        reference = pd.read_csv('tests/reference_frame.txt', delimiter=r'\s+', comment='#', header=None)
        traj = pd.read_csv('tests/example_rmsd_input.txt', delimiter=r'\s+', comment='#', header=None)
        num_atoms = traj.shape[1] // 3
        reference_frame = reference.iloc[0].to_numpy().reshape(-1, 3)
        rot = OptimalRotation(reference_frame)
        rmsds = []
        for i in range(0, traj.shape[0]):
            target_frame = traj.iloc[i].to_numpy().reshape(num_atoms, 3)
            rmsd = rot.optimal_rmsd(target_frame)
            rmsds.append(rmsd)
            print(f'Optimal RMSD between frame {i:4d} and reference is {rmsd:15.12f}')
        # compare the results with those from Colvars
        rmsds_colvars = pd.read_csv(
            'tests/example_rmsd_colvars.txt',
            delimiter=r'\s+', comment='#', header=None).iloc[:, 0].to_numpy()
        rmsds = np.asarray(rmsds)
        error = np.sqrt(np.mean(np.square(rmsds - rmsds_colvars)))
        print(f'Error = {error}')

    import argparse
    parser = argparse.ArgumentParser(description='Reading a colvars traj file line by line test')
    parser.add_argument('--test1', action='store_true', help='run rotation test')
    parser.add_argument('--test2', action='store_true', help='run optimal RMSD test')
    args = parser.parse_args()
    if args.test1:
        _test_rotation()
    if args.test2:
        _test_optimal_rmsd()
