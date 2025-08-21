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
from scipy import ndimage
from scipy.sparse import diags
from scipy.sparse import linalg as splinalg
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import scipy
import cv2
import vtk
import os
import time

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

    import numpy as np

    def robust_poisson_solver(self, normal_map, reg_strength=5.0, max_iter=3000, method='direct'):
        """
        鲁棒的Poisson方程求解法，同时解决内存和收敛性问题
        """
        h, w, _ = normal_map.shape
        print(f"处理图像尺寸: {h}x{w}, 总像素: {h*w}")
        
        # 转换法向量到[-1,1]范围
        if normal_map.max() > 1.0:
            normal_map = normal_map.astype(np.float32) / 255.0
        
        nx = 2.0 * normal_map[:, :, 0] - 1.0
        ny = 2.0 * normal_map[:, :, 1] - 1.0
        nz = normal_map[:, :, 2]
        
        # 归一化法向量
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        # 计算梯度场 - 添加稳定性处理
        p = -nx / (np.abs(nz) + 1e-6)
        q = -ny / (np.abs(nz) + 1e-6)
        
        # 限制梯度值范围，避免数值不稳定
        p = np.clip(p, -10, 10)
        q = np.clip(q, -10, 10)
        
        # 计算散度 - 使用稳定的有限差分
        divergence = np.zeros((h, w))
        
        # 使用中心差分计算散度（更稳定）
        for i in range(h):
            for j in range(w):
                # x方向导数
                if j == 0:
                    p_x = p[i, 1] - p[i, 0]
                elif j == w - 1:
                    p_x = p[i, w-1] - p[i, w-2]
                else:
                    p_x = (p[i, j+1] - p[i, j-1]) / 2.0
                
                # y方向导数
                if i == 0:
                    q_y = q[1, j] - q[0, j]
                elif i == h - 1:
                    q_y = q[h-1, j] - q[h-2, j]
                else:
                    q_y = (q[i+1, j] - q[i-1, j]) / 2.0
                
                divergence[i, j] = p_x + q_y
        
        # 根据图像大小自动选择方法
        total_pixels = h * w
        
        if method == 'auto':
            if total_pixels > 500000:
                method = 'multiscale'
            elif total_pixels > 100000:
                method = 'sparse'
            else:
                method = 'direct'
        
        print(f"选择求解方法: {method}")
        
        if method == 'multiscale':
            return self.multiscale_poisson_solver(normal_map, divergence)
        elif method == 'sparse':
            return self.sparse_poisson_solver(divergence, reg_strength, max_iter)
        else:  # direct
            return self.direct_poisson_solver(divergence, reg_strength)

    def sparse_poisson_solver(self, divergence, reg_strength=10.0, max_iter=5000):
        """
        使用稀疏矩阵的Poisson求解器（内存高效）
        """
        h, w = divergence.shape
        N = h * w
        
        print(f"构建稀疏矩阵系统，规模: {N}")
        
        # 使用更高效的稀疏矩阵构建方法
        # 每个点最多有5个非零元素（自身+4个邻居）
        nnz = 5 * N
        row_indices = np.zeros(nnz, dtype=np.int32)
        col_indices = np.zeros(nnz, dtype=np.int32)
        data_values = np.zeros(nnz, dtype=np.float32)
        
        idx = 0
        for i in range(h):
            for j in range(w):
                center_idx = i * w + j
                
                # 主对角线元素
                row_indices[idx] = center_idx
                col_indices[idx] = center_idx
                data_values[idx] = -4 * reg_strength
                idx += 1
                
                # 邻居元素
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_idx = ni * w + nj
                        row_indices[idx] = center_idx
                        col_indices[idx] = neighbor_idx
                        data_values[idx] = reg_strength
                        idx += 1
        
        # 修剪多余的零
        row_indices = row_indices[:idx]
        col_indices = col_indices[:idx]
        data_values = data_values[:idx]
        
        A = csr_matrix((data_values, (row_indices, col_indices)), shape=(N, N))
        b = -divergence.flatten().astype(np.float32)
        
        # 使用预处理器提高收敛性
        print("使用预处理器提高收敛性...")
        M = diags(1.0 / (A.diagonal() + 1e-6), 0, format='csr')
        
        # 尝试多种求解器
        solvers = [
            ('cg', lambda: splinalg.cg(A, b, M=M, maxiter=max_iter, atol=1e-4))
        ]
        
        # for solver_name, solver_func in solvers:
        #     try:
        #         print(f"尝试 {solver_name} 求解器...")
        #         z_flat, info = solver_func()
                
        #         if info == 0:
        #             print(f"{solver_name} 求解成功")
        #             depth_map = z_flat.reshape((h, w))
        #             return self.apply_depth_postprocessing(depth_map)
        #         else:
        #             print(f"{solver_name} 未收敛 (info={info})")
                    
        #     except Exception as e:
        #         print(f"{solver_name} 失败: {e}")
        
        print("所有迭代求解器失败，使用FFT方法")
        return self.fft_solver_single_channel(divergence)

    def direct_poisson_solver(self, divergence, reg_strength=5.0):
        """
        直接求解器，适用于小图像
        """
        h, w = divergence.shape
        N = h * w
        
        if N > 10000:
            print("图像太大，不适合直接求解，切换到稀疏方法")
            return self.sparse_poisson_solver(divergence, reg_strength)
        
        print("使用直接求解器...")
        
        # 构建矩阵
        A = np.zeros((N, N))
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                A[idx, idx] = -4 * reg_strength
                
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        nidx = ni * w + nj
                        A[idx, nidx] = reg_strength
        
        b = -divergence.flatten()
        
        try:
            # 使用SVD求解提高稳定性
            U, s, Vh = np.linalg.svd(A, full_matrices=False)
            s_inv = 1.0 / (s + 1e-10)  # 正则化
            z_flat = Vh.T @ (s_inv * (U.T @ b))
            depth_map = z_flat.reshape((h, w))
            return self.apply_depth_postprocessing(depth_map)
        except np.linalg.LinAlgError:
            print("SVD求解失败，使用伪逆")
            z_flat = np.linalg.pinv(A) @ b
            depth_map = z_flat.reshape((h, w))
            return self.apply_depth_postprocessing(depth_map)

    def multiscale_poisson_solver(self, normal_map, divergence, levels=3):
        """
        多尺度Poisson求解器，适用于大图像
        """
        h, w = divergence.shape
        print(f"使用多尺度方法，原始尺寸: {h}x{w}")
        
        # 创建金字塔
        pyramid = [divergence]
        sizes = [(h, w)]
        
        for level in range(1, levels):
            h_new = max(h // (2 ** level), 32)
            w_new = max(w // (2 ** level), 32)
            if h_new < 32 or w_new < 32:
                break
            resized = cv2.resize(pyramid[-1], (w_new, h_new), interpolation=cv2.INTER_AREA)
            pyramid.append(resized)
            sizes.append((h_new, w_new))
        
        print(f"多尺度层级: {[f'{h}x{w}' for h, w in sizes]}")
        
        # 在最粗糙尺度使用稳定方法
        coarsest_div = pyramid[-1]
        h_c, w_c = coarsest_div.shape
        
        print(f"在最粗糙尺度 {h_c}x{w_c} 求解...")
        depth_coarse = self.direct_poisson_solver(coarsest_div) if h_c * w_c < 10000 else self.sparse_poisson_solver(coarsest_div)
        
        # 逐步上采样和细化
        current_depth = depth_coarse
        
        for level in range(len(pyramid) - 2, -1, -1):
            h_target, w_target = sizes[level]
            print(f"上采样到 {h_target}x{w_target}...")
            
            # 上采样
            current_depth = cv2.resize(current_depth, (w_target, h_target), interpolation=cv2.INTER_CUBIC)
            
            # 在当前尺度进行细化
            current_depth = self.refine_depth(current_depth, pyramid[level], iterations=10)
        
        return self.apply_depth_postprocessing(current_depth)

    def refine_depth(self, depth, divergence, iterations=10, alpha=0.05):
        """
        深度图细化，使用稳定的迭代方法
        """
        h, w = depth.shape
        current_depth = depth.copy()
        
        for iter_num in range(iterations):
            # 计算当前深度图的拉普拉斯
            laplacian = np.zeros_like(current_depth)
            
            # 手动计算拉普拉斯（更稳定）
            for i in range(1, h-1):
                for j in range(1, w-1):
                    laplacian[i, j] = (current_depth[i-1, j] + current_depth[i+1, j] + 
                                    current_depth[i, j-1] + current_depth[i, j+1] - 
                                    4 * current_depth[i, j])
            
            # 边界处理
            laplacian[0, :] = laplacian[1, :]
            laplacian[-1, :] = laplacian[-2, :]
            laplacian[:, 0] = laplacian[:, 1]
            laplacian[:, -1] = laplacian[:, -2]
            
            # 更新深度图
            residual = divergence - laplacian
            current_depth += alpha * residual
            
            # 监控收敛
            max_residual = np.max(np.abs(residual))
            if iter_num % 5 == 0:
                print(f"  细化迭代 {iter_num+1}/{iterations}, 最大残差: {max_residual:.6f}")
        
        return current_depth

    def apply_depth_postprocessing(self, depth_map):
        """
        深度图后处理，提高质量
        """
        # 中值滤波去除噪声
        depth_map = cv2.medianBlur(depth_map.astype(np.float32), 3)
        
        # 双边滤波保持边缘
        depth_map = cv2.bilateralFilter(depth_map, 5, 0.1, 5)
        
        return self.normalize_depth(depth_map)

    def normalize_depth(self, depth_map):
        """归一化深度图"""
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        if abs(depth_max - depth_min) < 1e-8:
            return np.zeros_like(depth_map)
        return (depth_map - depth_min) / (depth_max - depth_min)

    def fft_solver_single_channel(self, divergence):
        """
        稳定的FFT求解器
        """
        h, w = divergence.shape
        
        # 使用FFT求解Poisson方程
        ky = 2 * np.pi * np.fft.fftfreq(h)[:, np.newaxis]
        kx = 2 * np.pi * np.fft.fftfreq(w)[np.newaxis, :]
        
        k_squared = kx**2 + ky**2
        k_squared[0, 0] = 1.0  # 避免除以零
        
        divergence_fft = np.fft.fft2(divergence)
        z_fft = divergence_fft / (-k_squared + 1e-8)
        z_fft[0, 0] = 0  # 设置DC分量为零
        
        depth_map = np.real(np.fft.ifft2(z_fft))
        return self.normalize_depth(depth_map)

    def process_image_with_fallback(self, output_path=None):
        """
        带 fallback 机制的图像处理主函数
        """
        print("=" * 50)
        print("开始处理法向量图...")
        
        # 加载图像
        normal_map = np.reshape(self.N, (self.height, self.width, 3))
        h, w, _ = normal_map.shape
        print(f"成功加载图像: {w}x{h}")
        
        # 如果图像太大，自动下采样
        max_pixels = 1000000  # 100万像素
        if h * w > max_pixels:
            scale = np.sqrt(max_pixels / (h * w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"图像太大，下采样到: {new_w}x{new_h}")
            normal_map = cv2.resize(normal_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 尝试多种方法，带有fallback机制
        methods = [
            ('鲁棒Poisson求解', lambda: self.robust_poisson_solver(normal_map)),
            # ('FFT求解', lambda: self.fft_solver_single_channel_from_normal(normal_map)),
            # ('快速近似', lambda: self.fast_approximation(normal_map))
        ]
        
        depth_map = None
        for method_name, method_func in methods:
            try:
                print(f"\n尝试 {method_name}...")
                start_time = time.time()
                depth_map = method_func()
                elapsed = time.time() - start_time
                print(f"{method_name} 成功，耗时: {elapsed:.2f}秒")
                break
            except Exception as e:
                print(f"{method_name} 失败: {e}")
                continue
        
        if depth_map is None:
            print("所有方法都失败！")
            return None
        
        # 保存结果
        if output_path:
            depth_uint8 = (depth_map * 255).astype(np.uint8)
            cv2.imwrite(output_path, depth_uint8)
            print(f"深度图已保存: {output_path}")
        
        self.depth = depth_map

    def fft_solver_single_channel_from_normal(self, normal_map):
        """从法向量图计算散度并使用FFT求解"""
        h, w, _ = normal_map.shape
        
        if normal_map.max() > 1.0:
            normal_map = normal_map.astype(np.float32) / 255.0
        
        nx = 2.0 * normal_map[:, :, 0] - 1.0
        ny = 2.0 * normal_map[:, :, 1] - 1.0
        nz = normal_map[:, :, 2]
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        p = -nx / (np.abs(nz) + 1e-6)
        q = -ny / (np.abs(nz) + 1e-6)
        p = np.clip(p, -5, 5)
        q = np.clip(q, -5, 5)
        
        p_x = np.gradient(p, axis=1)
        q_y = np.gradient(q, axis=0)
        divergence = p_x + q_y
        
        return self.fft_solver_single_channel(divergence)

    def fast_approximation(self, normal_map):
        """快速近似方法"""
        h, w, _ = normal_map.shape
        
        if normal_map.max() > 1.0:
            normal_map = normal_map.astype(np.float32) / 255.0
        
        # 简单使用Z分量作为深度基础
        depth = normal_map[:, :, 2].copy()
        
        # 多次高斯平滑
        for _ in range(3):
            depth = cv2.GaussianBlur(depth, (5, 5), 1.0)
        
        return self.normalize_depth(depth)


    def simple_depth_estimation(self):
        """
        简单的深度估计方法，作为备选方案
        """
        normal_map = np.reshape(self.N, (self.height, self.width, 3))
        h, w, _ = normal_map.shape
        
        if normal_map.max() > 1.0:
            normal_map = normal_map.astype(np.float32) / 255.0
        
        # 直接使用Z分量作为深度估计
        depth_map = normal_map[:, :, 2].copy()
        
        # 简单的积分方法
        depth_integrated = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                if i == 0 and j == 0:
                    depth_integrated[i, j] = depth_map[i, j]
                elif i == 0:
                    depth_integrated[i, j] = depth_integrated[i, j-1] + depth_map[i, j]
                elif j == 0:
                    depth_integrated[i, j] = depth_integrated[i-1, j] + depth_map[i, j]
                else:
                    depth_integrated[i, j] = (depth_integrated[i-1, j] + depth_integrated[i, j-1]) / 2 + depth_map[i, j]
        
        self.depth = (depth_integrated - depth_integrated.min()) / (depth_integrated.max() - depth_integrated.min() + 1e-8)

    def fft_solver(self):
        """
        使用FFT方法从法向量图生成深度图
        """
        normal_map = np.reshape(self.N, (self.height, self.width, 3))
        h, w, _ = normal_map.shape
        
        if normal_map.max() > 1.0:
            normal_map = normal_map.astype(np.float32) / 255.0
        
        nx = 2.0 * normal_map[:, :, 0] - 1.0
        ny = 2.0 * normal_map[:, :, 1] - 1.0
        nz = normal_map[:, :, 2]
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        # 更稳定的梯度计算
        p = -nx / (np.abs(nz) + 1e-6)
        q = -ny / (np.abs(nz) + 1e-6)
        p = np.clip(p, -5, 5)
        q = np.clip(q, -5, 5)
        
        # 计算散度
        p_x = np.gradient(p, axis=1, edge_order=2)
        q_y = np.gradient(q, axis=0, edge_order=2)
        divergence = p_x + q_y
        
        # 使用FFT求解Poisson方程
        ky = 2 * np.pi * np.fft.fftfreq(h)[:, np.newaxis]
        kx = 2 * np.pi * np.fft.fftfreq(w)[np.newaxis, :]
        
        k_squared = kx**2 + ky**2
        k_squared[0, 0] = 1.0  # 避免除以零
        
        # FFT求解
        divergence_fft = np.fft.fft2(divergence)
        z_fft = divergence_fft / (-k_squared + 1e-8)
        z_fft[0, 0] = 0  # 设置DC分量为零
        
        depth_map = np.real(np.fft.ifft2(z_fft))
        
        # 去除可能的虚部
        depth_map = np.real(depth_map)
        
        self.depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

    def iterative_solver(self, iterations=1000, alpha=0.05):
        """
        改进的迭代优化方法
        """
        normal_map = np.reshape(self.N, (self.height, self.width, 3))
        h, w, _ = normal_map.shape
        
        if normal_map.max() > 1.0:
            normal_map = normal_map.astype(np.float32) / 255.0
        
        nx = 2.0 * normal_map[:, :, 0] - 1.0
        ny = 2.0 * normal_map[:, :, 1] - 1.0
        nz = normal_map[:, :, 2]
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        
        # 目标梯度场
        p_target = -nx / (np.abs(nz) + 1e-6)
        q_target = -ny / (np.abs(nz) + 1e-6)
        p_target = np.clip(p_target, -5, 5)
        q_target = np.clip(q_target, -5, 5)
        
        # 初始化深度图
        depth = np.zeros((h, w))
        
        # 迭代优化
        for iteration in range(iterations):
            # 计算当前深度图的梯度
            z_x = np.gradient(depth, axis=1, edge_order=2)
            z_y = np.gradient(depth, axis=0, edge_order=2)
            
            # 计算梯度误差
            error_x = z_x - p_target
            error_y = z_y - q_target
            
            # 计算误差的散度（驱动更新）
            error_x_x = np.gradient(error_x, axis=1, edge_order=2)
            error_y_y = np.gradient(error_y, axis=0, edge_order=2)
            update = error_x_x + error_y_y
            
            # 自适应学习率
            current_alpha = alpha * (0.1 + 0.9 * (1 - iteration/iterations))
            
            # 更新深度图
            depth -= current_alpha * update
            
            # 每50次迭代显示进度
            if iteration % 50 == 0:
                print(f"迭代 {iteration}/{iterations}, 最大更新: {np.max(np.abs(update)):.6f}")
        
        self.depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    def computedepth2(self):
        print("Experimental")
        normal_map = np.reshape(self.N, (self.height, self.width, 3))
        h = normal_map.shape[0]
        w = normal_map.shape[1]
        self.pgrads = np.zeros((h, w), dtype=np.float32)
        self.qgrads = np.zeros((h, w), dtype=np.float32)
        self.pgrads[0:h, 0:w] = normal_map[:, :, 0] / normal_map[:, :, 2]
        self.qgrads[0:h, 0:w] = normal_map[:, :, 1] / normal_map[:, :, 2]
        Z = np.zeros((h, w), dtype=np.float32)
        A = np.array([(1, -1, 0),(1, 0, -1)], dtype=np.float32)
        hh = 10
        print(A)
        Apinv = np.linalg.pinv(A)
        print(Apinv)
        for i in range (0, h-1):
            for j in range (0, w-1):
                arr = np.array([-self.pgrads[i,j],-self.qgrads[i,j]], dtype=np.float32)
                temp = np.einsum('ji,i->j', Apinv,arr)
                temp = np.absolute(temp)*hh
                Z[i, j] = temp[0]
                Z[i + 1, j] = temp[1]
                Z[i, j + 1] = temp[2]
        #Z = cv.bitwise_not(Z)
        #self.Z = cv.normalize(Z, None, 0, 10, cv.NORM_MINMAX, cv.CV_32FC1)
        self.depth = np.clip(Z, -10, 10)
        Znorm = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Znorm', Znorm)
        cv2.imshow('Z', Z)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite(output_fold+'z_norm_2.png',Znorm)
        # cv2.imwrite(output_fold+'Z_2.png',Z)

    def compare_methods(self, normal_map_path=None):
        """
        比较不同方法的性能和质量
        """
        # 生成或加载法向量图
        if normal_map_path and os.path.exists(normal_map_path):
            normal_map = cv2.imread(normal_map_path)
            normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
            print(f"加载法向量图: {normal_map_path}")
        else:
            # 生成示例法向量图（简单平面）
            print("生成示例法向量图...")
            h, w = 128, 128  # 使用较小尺寸提高稳定性
            normal_map = np.ones((h, w, 3)) * 0.5
            normal_map[:, :, 2] = 1.0  # 指向正前方
            
            # 添加一些变化
            for i in range(h):
                for j in range(w):
                    if (i - h//2)**2 + (j - w//2)**2 < (min(h, w)//4)**2:
                        # 球体区域
                        dx = j - w//2
                        dy = i - h//2
                        r = np.sqrt(dx**2 + dy**2)
                        if r > 0:
                            normal_map[i, j, 0] = (dx/r + 1) / 2
                            normal_map[i, j, 1] = (dy/r + 1) / 2
                            normal_map[i, j, 2] = 0.7
        
        normal_map = (normal_map * 255).astype(np.uint8)
        
        # 运行不同方法
        methods = {
            "鲁棒Poisson求解": lambda x: self.robust_poisson_solver(x, reg_strength=5.0, max_iter=3000),
            "FFT求解": self.fft_solver,
            "迭代优化": lambda x: self.iterative_solver(x, iterations=300, alpha=0.03),
            "简单估计": self.simple_depth_estimation,
        }
        
        results = {}
        times = {}
        
        for name, method in methods.items():
            print(f"\n运行 {name}...")
            try:
                start_time = time.time()
                results[name] = method(normal_map.copy())
                times[name] = time.time() - start_time
                print(f"{name} 耗时: {times[name]:.3f}秒")
            except Exception as e:
                print(f"{name} 失败: {e}")
                results[name] = np.zeros(normal_map.shape[:2])
                times[name] = 0
        
        # 可视化结果
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(normal_map)
        plt.title('法向量图')
        plt.axis('off')
        
        for i, (name, depth) in enumerate(results.items(), 2):
            plt.subplot(2, 3, i)
            plt.imshow(depth, cmap='viridis')
            plt.title(f'{name}\n耗时: {times[name]:.3f}s')
            plt.axis('off')
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()
        
        return results, times


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
            if h >= im_h - 1 or w >= im_w - 1:
                continue
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

    def display3dobj(self, output_prefix="3d_model"):
        colors = vtk.vtkNamedColors()

        h, w = self.height, self.width

        # Create a triangle
        points = vtk.vtkPoints()
        for x in range(0, h):
            for y in range(0, w):
                points.InsertNextPoint(x, y, self.depth[x, y])

        triangle = vtk.vtkTriangle()
        triangles = vtk.vtkCellArray()
        for i in range(0, h-1):  # 修改边界条件，防止越界
            for j in range(0, w-1):
                # 第一个三角形
                triangle.GetPointIds().SetId(0, j + (i * w))
                triangle.GetPointIds().SetId(1, (i + 1) * w + j)
                triangle.GetPointIds().SetId(2, j + (i * w) + 1)
                triangles.InsertNextCell(triangle)
                # 第二个三角形
                triangle.GetPointIds().SetId(0, (i + 1)*w + j)
                triangle.GetPointIds().SetId(1, (i + 1)*w + j + 1)
                triangle.GetPointIds().SetId(2, j + (i*w) + 1)
                triangles.InsertNextCell(triangle)

        # Create a polydata object
        trianglePolyData = vtk.vtkPolyData()
        trianglePolyData.SetPoints(points)
        trianglePolyData.SetPolys(triangles)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(trianglePolyData)
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(colors.GetColor3d("Cyan"))
        actor.SetMapper(mapper)

        # Create a renderer, render window, and an interactor
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetWindowName("3D Surface Model")
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Add the actors to the scene
        renderer.AddActor(actor)
        renderer.SetBackground(colors.GetColor3d("DarkGreen"))

        # Render and interact
        renderWindow.Render()
        renderWindowInteractor.Start()

        # 保存OBJ文件
        obj_exporter = vtk.vtkOBJExporter()
        obj_exporter.SetFilePrefix(output_prefix)
        obj_exporter.SetInput(renderWindow)
        obj_exporter.Write()
        print(f"OBJ文件已保存到: {output_prefix}.obj 和相关文件")

        # 额外保存STL文件
        stl_writer = vtk.vtkSTLWriter()
        stl_filename = f"{output_prefix}.stl"
        stl_writer.SetFileName(stl_filename)
        stl_writer.SetInputData(trianglePolyData)
        stl_writer.Write()
        print(f"STL文件已保存到: {stl_filename}")

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