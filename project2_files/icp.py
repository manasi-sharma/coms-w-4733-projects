import argparse
import os

import cv2
import numpy as np
import trimesh

import image
from transforms import depth_to_point_cloud, transform_point3s
from camera import Camera, cam_view2pose

parser = argparse.ArgumentParser()
parser.add_argument('--val', action='store_true', help='pose estimation for validation set')
parser.add_argument('--test', action='store_true', help='pose estimation for test set')

LIST_OBJ_FOLDERNAME = [
        "004_sugar_box",  # obj_id == 1
        "005_tomato_soup_can",  # obj_id == 2
        "007_tuna_fish_can",  # obj_id == 3
        "011_banana",  # obj_id == 4
        "024_bowl",  # obj_id == 5
    ]


def obj_mesh2pts(obj_id, point_num, transform=None):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        point_num: int, number of points to sample.
        transform: Numpy array [4, 4] of float64.
    Out:
        pts: Numpy array [n, 3], sampled point cloud.
    Purpose:
         Sample a point cloud from the mesh of the specific object. If transform is not None, apply it.
    """
    mesh_path = './YCB_subsubset/' + LIST_OBJ_FOLDERNAME[obj_id - 1] + '/model_com.obj'  # objects ID start from 1
    mesh = trimesh.load(mesh_path)
    if transform is not None:
        mesh = mesh.apply_transform(transform)
    pts, _ = trimesh.sample.sample_surface(mesh, count=point_num)
    return pts


def gen_obj_depth(obj_id, depth, mask):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        obj_depth: Numpy array [height, width] of float64, where depth value of all the pixels that don't belong to the object is 0.
    Purpose:
        Generate depth image for a specific object.
        Generate depth for all objects when obj_id == -1.
    """
    # TODO
    h_len= len(mask)
    w_len= len(mask[0])

    obj_depth = np.copy(depth)

    if(obj_id != -1):

        for h in range(h_len):
            for w in range(w_len):
                mask_id= mask[h][w]

                if(obj_id != mask_id):
                    obj_depth[h][w]= 0

    else:

        for h in range(h_len):
            for w in range(w_len):
                mask_id= mask[h][w]

                if(mask_id == 0):
                    obj_depth[h][w]= 0

    return obj_depth


def obj_depth2pts(obj_id, depth, mask, camera, view_matrix):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        world_pts: Numpy array [n, 3], 3D points in the world frame of reference.
    Purpose:
        Generate point cloud projected from depth of the specific object(s) in the world frame of reference.
    Hint:
        The imported depth_to_point_cloud(), cam_view2pose() and transform_point3s() can be useful here.
        The view matrices are provided in the /dataset/val/view_matrix or /dataset/test/view_matrix folder.
    """
    # TODO
    obj_depth= gen_obj_depth(obj_id, depth, mask)
    camera_pts= depth_to_point_cloud(camera.intrinsic_matrix, obj_depth)

    #path= './dataset/val/view_matrix/'+str(obj_id)+'.npy'
    #cam_view_matrix= np.load(path)
    wTc= cam_view2pose(view_matrix)
    world_pts = transform_point3s(wTc, camera_pts)

    return world_pts


def align_pts(pts_a, pts_b, max_iterations=20, threshold=1e-05):
    """
    In:
        pts_a: Numpy array [n, 3].
        pts_b: Numpy array [n, 3].
        max_iterations: int, tunable parameter of trimesh.registration.icp().
        threshold: float，tunable parameter of trimesh.registration.icp().
    Out:
        matrix: Numpy array [4, 4], the transformation matrix sending pts_a to pts_b.
    Purpose:
        Apply the iterative closest point algorithm to estimate a transformation that aligns one point cloud with another.
    Hint:
        Use trimesh.registration.procrustes() and trimesh.registration.icp().
    """
    # TODO
    #t= 1e-05
    #m_i= 20
    if(len(pts_a)==0):
        return None

    """pts_a_trimesh= trimesh.Trimesh(pts_a)
    try:
        matrix, cost= trimesh.registration.mesh_other(pts_a_trimesh, pts_b, samples=300, scale=False)
    except np.linalg.LinAlgError:
        return None"""

    try:
        matrix, transformed, cost= trimesh.registration.procrustes(pts_a, pts_b, reflection=False, scale=False)
    except numpy.linalg.LinAlgError:
        return None
    #matrix= np.identity(4)
    #print("threshold: ", threshold)
    #icp_matrix, icp_transformed, icp_cost= trimesh.registration.icp(pts_a, pts_b, initial=matrix, threshold=threshold, max_iterations=max_iterations, samples=300, scale=False)
    icp_matrix, icp_transformed, icp_cost= trimesh.registration.icp(pts_a, pts_b, initial=matrix, threshold=threshold, max_iterations=max_iterations, reflection=False, scale=False)

    return icp_matrix


def estimate_pose(depth, mask, camera, view_matrix):
    """
    In:
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item should be None.
    Purpose:
        Perform pose estimation on each object in the given image.
    """
    # TODO
    list_obj_pose = list()

    # to change
    max_iterations= 40
    threshold= 1e-07
    #print("threshold: ", threshold)

    for i in range(5):
        proj_pnts= obj_depth2pts(i+1, depth, mask, camera, view_matrix)
        n= proj_pnts.shape[0]
        sample_pnts= obj_mesh2pts(i+1, n)
        #matrix= align_pts(sample_pnts, proj_pnts)
        matrix= align_pts(sample_pnts, proj_pnts, max_iterations=max_iterations, threshold=threshold)
        list_obj_pose.append(matrix)

    return list_obj_pose


def save_pose(dataset_dir, folder, scene_id, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        folder: string, the folder to save the pose.
                "gtmask" -- for pose estimated using ground truth mask
                "predmask" -- for pose estimated using predicted mask
        scene_id: int, ID of the scene.
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Save the pose of each object in a scene.
    """
    pose_dir = dataset_dir + "pred_pose/" + folder + "/"
    print(f"Save poses as .npy files to {pose_dir}")
    for i in range(len(list_obj_pose)):
        pose = list_obj_pose[i]
        if pose is not None:
            np.save(pose_dir + str(scene_id) + "_" + str(i + 1), pose)


def export_gt_ply(scene_id, depth, gt_mask, camera, view_matrix):
    """
    In:
        scene_id: int, ID of the scene.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        None.
    Purpose:
        Export a point cloud of the ground truth scene -- projected from depth using ground truth mask-- with the color green.
    """
    print("Export gt point cloud as .ply file to ./dataset/val/exported_ply/")
    file_path = "./dataset/val/exported_ply/" + str(scene_id) + "_gtmask.ply"
    pts = obj_depth2pts(-1, depth, gt_mask, camera, view_matrix)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(vertices=pts, colors=[0, 255, 0])  # Green
        ptcloud.export(file_path)


def export_pred_ply(dataset_dir, scene_id, suffix, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        scene_id: int, ID of the scene.
        suffix: string, indicating which kind of point cloud is going to be exported.
                "gtmask_transformed" -- transformed with pose estimated using ground truth mask
                "predmask_transformed" -- transformed with pose estimated using prediction mask
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Export a point cloud of the predicted scene with single color.
    """
    ply_dir = dataset_dir + "exported_ply/"
    print(f"Export predicted point cloud as .ply files to {ply_dir}")
    file_path = ply_dir + str(scene_id) + "_" + suffix + ".ply"
    color_switcher = {
        "gtmask_transformed": [0, 0, 255],  # Blue
        "predmask_transformed": [255, 0, 0],  # Red
    }
    pts = np.empty([0, 3])  # Numpy array [n, 3], the point cloud to be exported.
    for obj_id in range(1, 6):  # obj_id indicates an object in LIST_OBJ_FOLDERNAME
        pose = list_obj_pose[obj_id - 1]
        if pose is not None:
            obj_pts = obj_mesh2pts(obj_id, point_num=1000, transform=pose)
            pts = np.concatenate((pts, obj_pts), axis=0)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(vertices=pts, colors=color_switcher[suffix])
        ptcloud.export(file_path)


def denoise_mask(mask):
    """
    In:
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        mask: Numpy array [height, width], where each value is an obj_id, denoised.
    Purpose:
        Denoise the mask.
    Hint:
        Use dilate() and erode() method of OpenCV.
    """
    # TODO
    kernel = np.ones((5,5), np.uint8)
    mask= cv2.erode(mask, kernel, iterations = 1)
    mask= cv2.dilate(mask, kernel, iterations = 1)

    return mask


def main():
    args = parser.parse_args()
    if args.val:
        dataset_dir = "./dataset/val/"
        print("Pose estimation for validation set")
    elif args.test:
        dataset_dir = "./dataset/test/"
        print("Pose estimation for test set")
    else:
        print("Missing argument --val or --test")
        return

    # Setup camera -- to recover coordinate, keep consistency with that in gen_dataset.py
    my_camera = Camera(
        image_size=(240, 320),
        near=0.01,
        far=10.0,
        fov_w=69.40
    )

    if not os.path.exists(dataset_dir + "exported_ply/"):
        os.makedirs(dataset_dir + "exported_ply/")
    if not os.path.exists(dataset_dir + "pred_pose/"):
        os.makedirs(dataset_dir + "pred_pose/")
        os.makedirs(dataset_dir + "pred_pose/predmask/")
        if args.val:
            os.makedirs(dataset_dir + "pred_pose/gtmask/")

    # TODO:
    #  Use the implemented estimate_pose() to estimate the pose of the objects in each scene of the validation set and test set.
    #  For the validation set, use both ground truth mask and predicted mask.
    #  For the test set, use the predicted mask.
    #  Use save_pose(), export_gt_ply() and export_pred_ply() to generate files to be submitted.

    for scene_id in range(5):
        print("Estimating scene", scene_id)
        # TODO

        if args.val: #dataset_dir = "./dataset/val/"

            # gtmask -- val
            path= dataset_dir+'depth/'+str(scene_id)+'_depth.png'
            depth= image.read_depth(path)
            path= dataset_dir+'gt/'+str(scene_id)+'_gt.png'
            mask= image.read_mask(path)
            path= dataset_dir+'view_matrix/'+str(scene_id)+'.npy'
            view_matrix= np.load(path)
            export_gt_ply(scene_id, depth, mask, my_camera, view_matrix)

            # gtmask_transformed -- val
            list_obj_pose= estimate_pose(depth, mask, my_camera, view_matrix) #list of 5 matrices (each is 4x4 representing tranformations)
            export_pred_ply(dataset_dir, scene_id, 'gtmask_transformed', list_obj_pose)
            save_pose(dataset_dir, 'gtmask', scene_id, list_obj_pose)

            # predmask_transformed -- val
            path= dataset_dir+'pred/'+str(scene_id)+'_pred.png'
            mask= image.read_mask(path)
            list_obj_pose= estimate_pose(depth, mask, my_camera, view_matrix) #list of 5 matrices (each is 4x4 representing tranformations)
            export_pred_ply(dataset_dir, scene_id, 'predmask_transformed', list_obj_pose)
            save_pose(dataset_dir, 'predmask', scene_id, list_obj_pose)


        elif args.test: #dataset_dir = "./dataset/test/"
            # predmask_transformed -- test
            path= dataset_dir+'depth/'+str(scene_id)+'_depth.png'
            depth= image.read_depth(path)
            path= dataset_dir+'pred/'+str(scene_id)+'_pred.png'
            mask= image.read_mask(path)
            path= dataset_dir+'view_matrix/'+str(scene_id)+'.npy'
            view_matrix= np.load(path)
            list_obj_pose= estimate_pose(depth, mask, my_camera, view_matrix) #list of 5 matrices (each is 4x4 representing tranformations)
            export_pred_ply(dataset_dir, scene_id, 'predmask_transformed', list_obj_pose)
            save_pose(dataset_dir, 'predmask', scene_id, list_obj_pose)


if __name__ == '__main__':
    main()
