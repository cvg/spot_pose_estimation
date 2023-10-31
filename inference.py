from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
import os
import sys
from omegaconf import DictConfig
import yaml

# important for DeepLM module, this line should before import torch
os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"
import os.path as osp
import glob
import numpy as np
import natsort
import torch
import cv2

import hydra
# sys.path.append('/OnePose_Plus_Plus_Spot')
from src.utils import data_utils
from src.utils import vis_utils
from src.utils.data_io import read_grayscale
from src.inference.inference_OnePosePlus import build_model
from src.utils.metric_utils import ransac_PnP
from src.datasets.OnePosePlus_inference_dataset import OnePosePlusInferenceDataset
from src.local_feature_object_detector.local_feature_2D_detector import LocalFeatureObjectDetector
from demo import get_default_paths
from run import sfm


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    # main()

    with open('configs/experiment/inference_demo.yaml', 'r') as f:
        onepose_config = yaml.load(f, Loader=yaml.FullLoader)

    # NOTE: if you find pose estimation results are not good, problem maybe due to the poor object detection at the very beginning of the sequence.
    # You can set `output_results=True`, the detection results will thus be saved in the `detector_vis` directory in folder of the test sequence.
    img_list, paths = get_default_paths(
        {},
        data_root='/data/SpotRobot',
        data_dir='/data/SpotRobot/SpotRobot-annotate',
        sfm_model_dir=
        '/OnePose_Plus_Plus_Spot/data/demo/spot_model/outputs_softmax_loftr_loftr/SpotRobot',
    )
    # img_lists, paths = get_default_paths()

    local_feature_obj_detector = LocalFeatureObjectDetector(
        sfm_ws_dir=paths["sfm_ws_dir"],
        output_results=True,
        detect_save_dir=paths["vis_detector_dir"],
    )
    match_2D_3D_model = build_model(onepose_config['model']["OnePosePlus"],
                                    onepose_config['model']['pretrained_ckpt'])
    match_2D_3D_model.cuda()

    dataset = OnePosePlusInferenceDataset(
        paths['sfm_dir'],
        img_list,
        load_3d_coarse=True,
        shape3d=7000,
        img_pad=False,
        img_resize=None,
        df=8,
        pad=False,
        load_pose_gt=False,
        n_images=None,
        demo_mode=True,
        preload=True,
    )

    K, _ = data_utils.get_K(paths["intrin_full_path"])
    query_image_path = '/data/SpotRobot/SpotRobot-annotate/color/22.png'
    query_image, query_image_scale, query_image_mask = read_grayscale(
        query_image_path, resize=dataset.img_resize, pad_to=None, ret_scales=True, ret_pad_mask=True, df=dataset.df)
    bbox, inp_crop, K_crop = local_feature_obj_detector.detect(query_image, query_image_path, K)
    print(f"bbox: {bbox}")

    input_data = {
        "query_image": inp_crop.cuda(),
        "query_image_path": query_image_path,
        "descriptors3d_coarse_db": dataset.avg_coarse_descriptors3d[None],
        "descriptors3d_db": dataset.avg_descriptors3d[None],
        "keypoints3d": dataset.keypoints3d[None],
    }

    # Perform keypoint-free 2D-3D matching and then estimate object pose of query image by PnP:
    with torch.no_grad():
        match_2D_3D_model(input_data)
    mkpts_3d = input_data["mkpts_3d_db"].cpu().numpy() # N*3
    mkpts_query = input_data["mkpts_query_f"].cpu().numpy() # N*2
    pose_pred, _, inliers, _ = ransac_PnP(K_crop, mkpts_query, mkpts_3d, scale=1000, pnp_reprojection_error=7, img_hw=[512,512], use_pycolmap_ransac=True)
    print(f"pose_pred: {pose_pred}, inliers: {inliers}")
    # Visualize:
    bbox3d = np.loadtxt(paths["bbox3d_path"])
    vis_utils.save_demo_image(
        pose_pred,
        K,
        image_path=query_image_path,
        box3d=bbox3d,
        draw_box=len(inliers) > 20,
        save_path="outputs/pose_pred.jpg",
    )

