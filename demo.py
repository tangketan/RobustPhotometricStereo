from __future__ import print_function

import numpy as np
import time
from rps import RPS
import psutils
# import open3d as o3d

# Choose a method
# METHOD = RPS.L2_SOLVER    # Least-squares
# METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
#METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
METHOD = RPS.RPCA_SOLVER    # Robust PCA

use_bunny=0
if use_bunny:
    # Choose a dataset
    #DATA_FOLDERNAME = './data/bunny/bunny_specular/'    # Specular with cast shadow
    DATA_FOLDERNAME = './data/bunny/bunny_lambert/'    # Lambertian diffuse with cast shadow
    #DATA_FOLDERNAME = './data/bunny/bunny_lambert_noshadow/'    # Lambertian diffuse without cast shadow
    LIGHT_FILENAME = './data/bunny/lights.npy'
    MASK_FILENAME = './data/bunny/mask.png'
    GT_NORMAL_FILENAME = './data/bunny/gt_normal.npy'
else:
    DATA_FOLDERNAME = './data/render_output/'
    MASK_FILENAME = DATA_FOLDERNAME + 'mask.png'
    GT_NORMAL_FILENAME = None#DATA_FOLDERNAME + 'normal_true.png'

# Photometric Stereo
rps = RPS()
if use_bunny:
    rps.load_mask(filename=MASK_FILENAME)    # Load mask image
    rps.load_lightnpy(filename=LIGHT_FILENAME)    # Load light matrix
    rps.load_images(foldername=DATA_FOLDERNAME, ext="npy")    # Load observations
else:
    rps.load_mask(filename=MASK_FILENAME, background_is_zero=True)    # Load mask image
    rps.load_light_yaml(folder_name=DATA_FOLDERNAME)
    rps.load_images(foldername=DATA_FOLDERNAME+"images/", ext="png")
start = time.time()
rps.solve(METHOD)    # Compute
elapsed_time = time.time() - start
print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
rps.save_normalmap(filename="./est_normal")    # Save the estimated normal map

# Evaluate the estimate
if GT_NORMAL_FILENAME is not None:
    N_gt = psutils.load_normalmap_from_npy(filename=GT_NORMAL_FILENAME)    # read out the ground truth surface normal
    N_gt = np.reshape(N_gt, (rps.height*rps.width, 3))    # reshape as a normal array (p \times 3)
    angular_err = psutils.evaluate_angular_error(N_gt, rps.N, rps.background_ind)    # compute angular error
    print("Mean angular error [deg]: ", np.mean(angular_err[:]))
    psutils.disp_normalmap(normal=rps.N.copy(), height=rps.height, width=rps.width)
print("done.")

# 计算出深度图，参考：https://blog.csdn.net/SZU_Kwong/article/details/112757354
rps.compute_depth()
rps.save_depthmap(filename="./est_depth")
psutils.disp_depthmap(depth=rps.depth, mask=rps.mask)

rps.get_mesh("./est_mesh.stl")

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.array(points))
# pcd.normals = o3d.utility.Vector3dVector(np.array(normal))
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)
# o3d.io.write_point_cloud("./est_pointcloud.pcd", pcd)