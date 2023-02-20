import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import torch
import numpy as np
import sys
import os
from fusion import *

BASE_DIR = os.getcwd().rsplit("/", 1)[0]
sys.path.append(BASE_DIR)
print("adding Base Dir Path", BASE_DIR)
# import fusion


def read_calib(calib_path):
    """
    Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if line == "\n":
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    # 3x4 projection matrix for left camera
    calib_out["P2"] = calib_all["P2"].reshape(3, 4)
    calib_out["Tr"] = np.identity(4)  # 4x4 matrix
    calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
    return calib_out


def vox2pix(cam_E, cam_k,
            vox_origin, voxel_size,
            img_W, img_H,
            scene_size):
    """
    compute the 2D projection of voxels centroids

    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV 
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = vox_origin
    vol_bnds[:, 1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) /
                      voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
        range(vol_dim[0]),
        range(vol_dim[1]),
        range(vol_dim[2]),
        indexing='ij'
    )
    vox_coords = np.concatenate([
        xv.reshape(1, -1),
        yv.reshape(1, -1),
        zv.reshape(1, -1)
    ], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                              np.logical_and(pix_x < img_W,
                                             np.logical_and(pix_y >= 0,
                                                            np.logical_and(pix_y < img_H,
                                                                           pix_z > 0))))

    return torch.from_numpy(projected_pix), torch.from_numpy(fov_mask), torch.from_numpy(pix_z)


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def get_projections(img_W, img_H):
    scale_3ds = [1, 2]
    data = {}
    for scale_3d in scale_3ds:
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = 0.2

        calib = read_calib("/monoscene/MonoScene/calib.txt")
        cam_k = calib["P2"][:3, :3]
        T_velo_2_cam = calib["Tr"]

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(
            T_velo_2_cam,
            cam_k,
            vox_origin,
            voxel_size * scale_3d,
            img_W,
            img_H,
            scene_size,
        )

        data["projected_pix_{}".format(scale_3d)] = projected_pix
        data["pix_z_{}".format(scale_3d)] = pix_z
        data["fov_mask_{}".format(scale_3d)] = fov_mask
    return data


def majority_pooling(grid, k_size=2):
    result = np.zeros(
        (grid.shape[0] // k_size, grid.shape[1] //
         k_size, grid.shape[2] // k_size)
    )
    for xx in range(0, int(np.floor(grid.shape[0] / k_size))):
        for yy in range(0, int(np.floor(grid.shape[1] / k_size))):
            for zz in range(0, int(np.floor(grid.shape[2] / k_size))):

                sub_m = grid[
                    (xx * k_size): (xx * k_size) + k_size,
                    (yy * k_size): (yy * k_size) + k_size,
                    (zz * k_size): (zz * k_size) + k_size,
                ]
                unique, counts = np.unique(sub_m, return_counts=True)
                if True in ((unique != 0) & (unique != 255)):
                    # Remove counts with 0 and 255
                    counts = counts[((unique != 0) & (unique != 255))]
                    unique = unique[((unique != 0) & (unique != 255))]
                else:
                    if True in (unique == 0):
                        counts = counts[(unique != 255)]
                        unique = unique[(unique != 255)]
                value = unique[np.argmax(counts)]
                result[xx, yy, zz] = value
    return result


def draw(
    voxels,
    # T_velo_2_cam,
    # vox_origin,
    fov_mask,
    # img_size,
    # f,
    voxel_size=0.4,
    # d=7,  # 7m - determine the size of the mesh representing the camera
):

    fov_mask = fov_mask.reshape(-1)
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255), :
    ]
    # print(np.unique(fov_voxels[:, 3], return_counts=True))
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255), :
    ]

    # figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    colors = np.array(
        [
            [0, 0, 0],
            [100, 150, 245],
            [100, 230, 245],
            [30, 60, 150],
            [80, 30, 180],
            [100, 80, 250],
            [255, 30, 30],
            [255, 40, 200],
            [150, 30, 90],
            [255, 0, 255],
            [255, 150, 255],
            [75, 0, 75],
            [175, 0, 75],
            [255, 200, 0],
            [255, 120, 50],
            [0, 175, 0],
            [135, 60, 0],
            [150, 240, 80],
            [255, 240, 150],
            [255, 0, 0],
        ]
    ).astype(np.uint8)

    pts_colors = [
        f'rgb({colors[int(i)][0]}, {colors[int(i)][1]}, {colors[int(i)][2]})' for i in fov_voxels[:, 3]]
    out_fov_colors = [
        f'rgb({colors[int(i)][0]//3*2}, {colors[int(i)][1]//3*2}, {colors[int(i)][2]//3*2})' for i in outfov_voxels[:, 3]]
    pts_colors = pts_colors + out_fov_colors

    fov_voxels = np.concatenate([fov_voxels, outfov_voxels], axis=0)
    x = fov_voxels[:, 0].flatten()
    y = fov_voxels[:, 1].flatten()
    z = fov_voxels[:, 2].flatten()
    # label = fov_voxels[:, 3].flatten()
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                    marker=dict(
                        size=2,
                        color=pts_colors,                # set color to an array/list of desired values
                        # colorscale='Viridis',   # choose a colorscale
                        opacity=1.0,
                        symbol='square'
                    ))])
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(
                backgroundcolor="rgb(255, 255, 255)",
                gridcolor="black",
                showbackground=True,
                zerolinecolor="black",
                nticks=4,
                visible=False,
                range=[-1, 55],),
            yaxis=dict(
                backgroundcolor="rgb(255, 255, 255)",
                gridcolor="black",
                showbackground=True,
                zerolinecolor="black",
                visible=False,
                nticks=4, range=[-1, 55],),
            zaxis=dict(
                backgroundcolor="rgb(255, 255, 255)",
                gridcolor="black",
                showbackground=True,
                zerolinecolor="black",
                visible=False,
                nticks=4, range=[-1, 7],),
            bgcolor="black",
        ),

    )

    return fig
