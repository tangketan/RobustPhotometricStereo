#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Photometric Stereo in Python
"""
__author__ = "Yasuyuki Matsushita <yasumat@ist.osaka-u.ac.jp>"
__version__ = "0.1.0"
__date__ = "11 May 2018"

import psutils
import rpsnumerics
import numpy as np
from sklearn.preprocessing import normalize
import scipy
import cv2

class RPS(object):
    """
    Robust Photometric Stereo class
    """
    # Choice of solution methods
    L2_SOLVER = 0   # Conventional least-squares
    L1_SOLVER = 1   # L1 residual minimization
    L1_SOLVER_MULTICORE = 2 # L1 residual minimization (multicore)
    SBL_SOLVER = 3  # Sparse Bayesian Learning
    SBL_SOLVER_MULTICORE = 4    # Sparse Bayesian Learning (multicore)
    RPCA_SOLVER = 5    # Robust PCA

    def __init__(self):
        self.M = None   # measurement matrix in numpy array
        self.L = None   # light matrix in numpy array
        self.N = None   # surface normal matrix in numpy array
        self.height = None  # image height
        self.width = None   # image width
        self.mask = None  # mask image (numpy array)
        self.foreground_ind = None    # mask (indices of active pixel locations (rows of M))
        self.background_ind = None    # mask (indices of inactive pixel locations (rows of M))

    def load_lighttxt(self, filename=None):
        """
        Load light file specified by filename.
        The format of lights.txt should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.txt
        """
        self.L = psutils.load_lighttxt(filename)

    def load_lightnpy(self, filename=None):
        """
        Load light numpy array file specified by filename.
        The format of lights.npy should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.npy
        """
        self.L = psutils.load_lightnpy(filename)

    def load_light_yaml(self, folder_name=None):
        fs = cv2.FileStorage(folder_name + "LightMatrix.yml", cv2.FILE_STORAGE_READ)
        fn = fs.getNode("Lights")
        self.L = fn.mat().T

    def load_images(self, foldername=None, ext='npy'):
        """
        Load images in the folder specified by the "foldername" in the numpy format
        :param foldername: foldername
        """
        if ext=='npy':
            self.M, self.height, self.width = psutils.load_npyimages(foldername)
        else:
            self.M, self.height, self.width = psutils.load_images(foldername, ext)

    def load_mask(self, filename=None, background_is_zero=True):
        """
        Load mask image and set the mask indices
        In the mask image, pixels with zero intensity will be ignored.
        :param filename: filename of the mask image
        :return: None
        """
        if filename is None:
            raise ValueError("filename is None")
        mask = psutils.load_image(filename=filename)
        if not background_is_zero:
            mask = 255 - mask
        self.mask = mask.copy()
        mask = mask.reshape((-1, 1))
        self.foreground_ind = np.where(mask != 0)[0]
        self.background_ind = np.where(mask == 0)[0]

    def disp_normalmap(self, delay=0):
        """
        Visualize normal map
        :return: None
        """
        psutils.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=delay)

    def save_normalmap(self, filename=None):
        """
        Saves normal map as numpy array format (npy)
        :param filename: filename of a normal map
        :return: None
        """
        psutils.save_normalmap_as_npy(filename=filename, normal=self.N, height=self.height, width=self.width)

    def solve(self, method=L2_SOLVER):
        if self.M is None:
            raise ValueError("Measurement M is None")
        if self.L is None:
            raise ValueError("Light L is None")
        if self.M.shape[1] != self.L.shape[1]:
            raise ValueError("Inconsistent dimensionality between M and L")

        if method == RPS.L2_SOLVER:
            self._solve_l2()
        elif method == RPS.L1_SOLVER:
            self._solve_l1()
        elif method == RPS.L1_SOLVER_MULTICORE:
            self._solve_l1_multicore()
        elif method == RPS.SBL_SOLVER:
            self._solve_sbl()
        elif method == RPS.SBL_SOLVER_MULTICORE:
            self._solve_sbl_multicore()
        elif method == RPS.RPCA_SOLVER:
            self._solve_rpca()
        else:
            raise ValueError("Undefined solver")

    def _solve_l2(self):
        """
        Lambertian Photometric stereo based on least-squares
        Woodham 1980
        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        self.N = np.linalg.lstsq(self.L.T, self.M.T, rcond=None)[0].T
        self.N = normalize(self.N, axis=1)  # normalize to account for diffuse reflectance
        if self.background_ind is not None:
            for i in range(self.N.shape[1]):
                self.N[self.background_ind, i] = 0

    def _solve_l1(self):
        """
        Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        A = self.L.T
        self.N = np.zeros((self.M.shape[0], 3))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind

        for index in indices:
            b = np.array([self.M[index, :]]).T
            n = rpsnumerics.L1_residual_min(A, b)
            self.N[index, :] = n.ravel()
        self.N = normalize(self.N, axis=1)

    def _solve_l1_multicore(self):
        """
        Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        from multiprocessing import Pool
        import multiprocessing

        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        p = Pool(processes=multiprocessing.cpu_count()-1)
        normal = p.map(self._solve_l1_multicore_impl, indices)
        if self.foreground_ind is None:
            self.N = np.asarray(normal)
            self.N = normalize(self.N, axis=1)
        else:
            N = np.asarray(normal)
            N = normalize(N, axis=1)
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]

    def _solve_l1_multicore_impl(self, index):
        """
        Implementation of Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :param index: an index of a measurement (row of M)
        :return: a row vector of surface normal at pixel index specified by "index"
        """
        A = self.L.T
        b = np.array([self.M[index, :]]).T
        n = rpsnumerics.L1_residual_min(A, b)   # row vector of a surface normal at pixel "index"
        return n.ravel()

    def _solve_sbl(self):
        """
        Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        A = self.L.T
        self.N = np.zeros((self.M.shape[0], 3))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind

        for index in indices:
            b = np.array([self.M[index, :]]).T
            n = rpsnumerics.sparse_bayesian_learning(A, b)
            self.N[index, :] = n.ravel()
        self.N = normalize(self.N, axis=1)

    def _solve_sbl_multicore(self):
        """
        Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        from multiprocessing import Pool
        import multiprocessing

        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        p = Pool(processes=multiprocessing.cpu_count()-1)
        normal = p.map(self._solve_sbl_multicore_impl, indices)
        if self.foreground_ind is None:
            self.N = np.asarray(normal)
            self.N = normalize(self.N, axis=1)
        else:
            N = np.asarray(normal)
            N = normalize(N, axis=1)
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(self.N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]

    def _solve_sbl_multicore_impl(self, index):
        """
        Implementation of Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :param index: an index of a measurement (row of M)
        :return: a row vector of surface normal at pixel index specified by "index"
        """
        A = self.L.T
        b = np.array([self.M[index, :]]).T
        n = rpsnumerics.sparse_bayesian_learning(A, b)   # row vector of a surface normal at pixel "index"
        return n.ravel()

    def _solve_rpca(self):
        """
        Photometric stereo based on robust PCA.
        Lun Wu, Arvind Ganesh, Boxin Shi, Yasuyuki Matsushita, Yongtian Wang, Yi Ma:
        Robust Photometric Stereo via Low-Rank Matrix Completion and Recovery. ACCV (3) 2010: 703-717

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        if self.foreground_ind is None:
            _M = self.M.T
        else:
            _M = self.M[self.foreground_ind, :].T

        A, E, ite = rpsnumerics.rpca_inexact_alm(_M)    # RPCA Photometric stereo

        if self.foreground_ind is None:
            self.N = np.linalg.lstsq(self.L.T, A, rcond=None)[0].T
            self.N = normalize(self.N, axis=1)    # normalize to account for diffuse reflectance
        else:
            N = np.linalg.lstsq(self.L.T, A, rcond=None)[0].T
            N = normalize(N, axis=1)    # normalize to account for diffuse reflectance
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(self.N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]


    def compute_depth(self):
        """
        计算出深度图
        原理参考:
        Mz = v
        M shape(2*numpixel, numpixel)
        z shape(numpixel, 1)
        v shape(2*numpixel, 1)
        1.http://pages.cs.wisc.edu/~csverma/CS766_09/Stereo/stereo.html
        2.https://www.zhihu.com/question/388447602/answer/1200616778
        """

        im_h, im_w = self.mask.shape
        N = np.reshape(self.N, (self.height, self.width, 3))

        # 得到掩膜图像非零值索引（即物体区域的索引）
        obj_h, obj_w = np.where(self.mask != 0)
        # 得到非零元素的数量
        no_pix = np.size(obj_h)
        # 构建一个矩阵 里面的元素值是掩膜图像索引的值
        full2obj = np.zeros((im_h, im_w), np.int32)
        for idx in range(np.size(obj_h)):
            full2obj[obj_h[idx], obj_w[idx]] = idx

        # Mz = v
        M = scipy.sparse.lil_matrix((2*no_pix, no_pix))
        v = np.zeros((2*no_pix, 1))

        #--------- 填充M和v -----------#
        # failed_rows = []
        for idx in range(no_pix):
            # 获取2D图像上的坐标
            h = obj_h[idx]
            w = obj_w[idx]
            # 获取表面法线
            n_x = N[h, w, 0]
            n_y = N[h, w, 1]
            n_z = N[h, w, 2]

            # z_(x+1, y) - z(x, y) = -nx / nz
            row_idx = idx * 2
            if self.mask[h, w+1]:
                idx_horiz = full2obj[h, w+1]
                M[row_idx, idx] = -1
                M[row_idx, idx_horiz] = 1
                v[row_idx] = -n_x / n_z
            elif self.mask[h, w-1]:
                idx_horiz = full2obj[h, w-1]
                M[row_idx, idx_horiz] = -1
                M[row_idx, idx] = 1
                v[row_idx] = -n_x / n_z
            # else:
            #     failed_rows.append(row_idx)

            # z_(x, y+1) - z(x, y) = -ny / nz
            row_idx = idx * 2 + 1
            if self.mask[h+1, w]:
                idx_vert = full2obj[h+1, w]
                M[row_idx, idx] = 1
                M[row_idx, idx_vert] = -1
                v[row_idx] = -n_y / n_z
            elif self.mask[h-1, w]:
                idx_vert = full2obj[h-1, w]
                M[row_idx, idx_vert] = 1
                M[row_idx, idx] = -1
                v[row_idx] = -n_y / n_z
            # else:
            #     failed_rows.append(row_idx)

        # # 将全零的行删除 对于稀疏矩阵M，要先将其恢复成稠密矩阵进行行删除再转为稀疏矩阵
        # M = M.todense()
        # M = np.delete(M, failed_rows, 0)
        # M = scipy.sparse.lil_matrix(M)
        # v = np.delete(v, failed_rows, 0)

        # 求解线性方程组 Mz = v  <<-->> M.T M z= M.T v
        MtM = M.T @ M
        Mtv = M.T @ v
        z = scipy.sparse.linalg.spsolve(MtM, Mtv)

        std_z = np.std(z, ddof=1)
        mean_z = np.mean(z)
        z_zscore = (z - mean_z) / std_z

        # 因奇异值造成的异常
        outlier_ind = np.abs(z_zscore) > 10
        z_min = np.min(z[~outlier_ind])
        z_max = np.max(z[~outlier_ind])

        # 将z填充回正常的2D形状
        Z = self.mask.astype('float')
        self.points = []
        for idx in range(no_pix):
            # 2D图像中的位置
            h = obj_h[idx]
            w = obj_w[idx]
            Z[h, w] = (z[idx] - z_min) / (z_max - z_min) * 255
            self.points.append((w, h, Z[h, w]))

        self.depth = Z


    def save_depthmap(self, filename=None):
        """
        将深度图保存为npy格式
        :param filename: filename of a depth map
        :retur: None
        """
        psutils.save_depthmap_as_npy(filename=filename, depth=self.depth)
        psutils.save_depth_and_normal_as_obj(filename=filename+".obj", depth=self.depth, normal=self.N, mask=self.mask)

    def get_mesh(self, filename=None):
        # compute 3d mesh from point cloud
        import open3d as o3d
        from scipy.spatial import Delaunay
        points = np.array(self.points)  # shape (N, 3)
        # Use only x, y for triangulation
        xy = points[:, :2]
        tri = Delaunay(xy)
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
        # Optionally assign normals if available
        if hasattr(self, 'N') and self.N is not None:
            normals = self.N.reshape(self.height, self.width, 3)
            # Map normals to valid points
            mesh.vertex_normals = o3d.utility.Vector3dVector([
                normals[int(p[1]), int(p[0])] for p in points
            ])
        mesh.compute_vertex_normals()
        if filename is not None:
            o3d.io.write_triangle_mesh(filename, mesh)
        return mesh