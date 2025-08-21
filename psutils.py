# psutils.py
import cv2
import glob
import numpy as np


def load_lighttxt(filename=None):
    """
    Load light file specified by filename.
    The format of lights.txt should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.txt
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.loadtxt(filename)
    return Lt.T


def load_lightnpy(filename=None):
    """
    Load light numpy array file specified by filename.
    The format of lights.npy should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.npy
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.load(filename)
    return Lt.T


def load_image(filename=None):
    """
    Load image specified by filename (read as a gray-scale)
    :param filename: filename of the image to be loaded
    :return img: loaded image
    """
    if filename is None:
        raise ValueError("filename is None")
    return cv2.imread(filename, 0)


def load_images(foldername=None, ext=None):
    """
    Load images in the folder specified by the "foldername" that have extension "ext"
    :param foldername: foldername
    :param ext: file extension
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None or ext is None:
        raise ValueError("filename/ext is None")

    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*." + ext)):
        im = cv2.imread(fname).astype(np.float64)
        if im.ndim == 3:
            # Assuming that RGBA will not be an input
            im = np.mean(im, axis=2)   # RGB -> Gray
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    return M, height, width


def load_npyimages(foldername=None):
    """
    Load images in the folder specified by the "foldername" in the numpy format
    :param foldername: foldername
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None:
        raise ValueError("filename is None")

    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*.npy")):
        im = np.load(fname)
        if im.ndim == 3:
            im = np.mean(im, axis=2)
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
        # cv2.imshow(fname, im)
        # cv2.waitKey(0)
    return M, height, width


def disp_normalmap(normal=None, height=None, width=None, delay=0, name=None, path=None):
    """
    Visualize normal as a normal map
    :param normal: array of surface normal (p \times 3)
    :param height: height of the image (scalar)
    :param width: width of the image (scalar)
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :param name: display name
    :return: None
    """
    if normal is None:
        raise ValueError("Surface normal `normal` is None")
    N = np.reshape(normal, (height, width, 3))  # Reshape to image coordinates
    N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()  # Swap RGB <-> BGR
    N = (N + 1.0) / 2.0  # Rescale
    if name is None:
        name = 'normal map'
    cv2.imshow(name, N)
    cv2.waitKey(delay)
    if path is not None:
        cv2.imwrite(path+name + ".png", N)
    # cv2.destroyWindow(name)
    # cv2.waitKey(1)    # to deal with frozen window...


def save_normalmap_as_npy(filename=None, normal=None, height=None, width=None):
    """
    Save surface normal array as a numpy array
    :param filename: filename of the normal array
    :param normal: surface normal array (height \times width \times 3)
    :return: None
    """
    if filename is None:
        raise ValueError("filename is None")
    N = np.reshape(normal, (height, width, 3))
    np.save(filename, N)


def load_normalmap_from_npy(filename=None):
    """
    Load surface normal array (which is a numpy array)
    :param filename: filename of the normal array
    :return: surface normal (numpy array) in formatted in (height, width, 3).
    """
    if filename is None:
        raise ValueError("filename is None")
    return np.load(filename)


def evaluate_angular_error(gtnormal=None, normal=None, background=None):
    if gtnormal is None or normal is None:
        raise ValueError("surface normal is not given")
    ae = np.multiply(gtnormal, normal)
    aesum = np.sum(ae, axis=1)
    coord = np.where(aesum > 1.0)
    aesum[coord] = 1.0
    coord = np.where(aesum < -1.0)
    aesum[coord] = -1.0
    ae = np.arccos(aesum) * 180.0 / np.pi
    if background is not None:
        ae[background] = 0
    return ae

def save_depthmap_as_npy(filename=None, depth=None):
    """
    将深度图保存为npy
    :param filename: filename of the depth array
    :param normal: surface depth array
    :return: None
    """
    if filename is None:
        raise ValueError("filename is None")
    np.save(filename, depth)

def save_depthmap_as_obj(filename=None, depth=None):
    """
    将深度图保存为obj格式
    :param filename: filename of the depth array
    :param depth: surface depth array
    :return: None
    """
    if filename is None:
        raise ValueError("filename is None")
    if depth is None:
        raise ValueError("depth is None")

    with open(filename, 'w') as f:
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                f.write("v {} {} {}\n".format(i, j, depth[i, j]))
        f.write("end\n")

def save_depth_and_normal_as_obj(filename=None, depth=None, normal=None, mask=None):
    """
    Save depth and normal maps as OBJ format with both vertices and normals.
    :param filename: output OBJ filename
    :param depth: (H, W) depth map
    :param normal: (H, W, 3) normal map
    :param mask: (H, W) optional mask, only save valid points
    """
    if filename is None:
        raise ValueError("filename is None")
    if depth is None or normal is None:
        raise ValueError("depth or normal is None")

    H, W = depth.shape
    # convert normal to H x W
    normal_reshaped = normal.reshape((H, W, 3))
    with open(filename, 'w') as f:
        # Write vertices and normals
        for i in range(H):
            for j in range(W):
                if mask is not None and mask[i, j] == 0:
                    continue
                z = depth[i, j]
                nx, ny, nz = normal_reshaped[i, j]
                f.write(f"vn {nx:.7f} {ny:.7f} {nz:.7f}\n")
                f.write(f"v {j} {i} {z:.7f}\n")
        # Optionally, faces can be added if mesh connectivity is known


def disp_depthmap(depth=None, mask=None, delay=0, name=None, path=None):
    """
    显示深度图
    :param depth: array of surface depth
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :param name: display name
    :return: None
    """
    if depth is None:
        raise ValueError("Surface depth `depth` is None")
    if mask is not None:
        depth = depth * mask

    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)  # Rescale to [0, 255]
    depth = np.uint8(depth)

    if name is None:
        name = 'depth map'

    if path is not None:
        cv2.imwrite(path+name + ".png", depth)

    cv2.imshow(name, depth)
    cv2.waitKey(delay)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
