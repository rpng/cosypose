from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import time
import json

from collections import OrderedDict
import yaml
import argparse

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import logging

from cosypose.config import EXP_DIR, MEMORY, RESULTS_DIR, LOCAL_DATA_DIR

from cosypose.utils.distributed import init_distributed_mode, get_world_size, get_rank
from cosypose.datasets.samplers import DistributedSceneSampler

from cosypose.lib3d import Transform

from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse, check_update_config
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor

from cosypose.evaluation.meters.pose_meters import PoseErrorMeter
from cosypose.evaluation.pred_runner.multiview_predictions import MultiviewPredictionRunner
from cosypose.evaluation.eval_runner.pose_eval import PoseEvaluation

import cosypose.utils.tensor_collection as tc
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from cosypose.utils.distributed import get_rank


from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.datasets.bop import remap_bop_targets
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

from cosypose.datasets.samplers import ListSampler
from cosypose.utils.logging import get_logger
logger = get_logger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@MEMORY.cache
def load_posecnn_results():
    results_path = LOCAL_DATA_DIR / 'saved_detections' / 'ycbv_posecnn.pkl'
    results = pkl.loads(results_path.read_bytes())
    infos, poses, bboxes = [], [], []

    l_offsets = (LOCAL_DATA_DIR / 'bop_datasets/ycbv' / 'offsets.txt').read_text().strip().split('\n')
    ycb_offsets = dict()
    for l_n in l_offsets:
        obj_id, offset = l_n[:2], l_n[3:]
        obj_id = int(obj_id)
        offset = np.array(json.loads(offset)) * 0.001
        ycb_offsets[obj_id] = offset

    def mat_from_qt(qt):
        wxyz = qt[:4].copy().tolist()
        xyzw = [*wxyz[1:], wxyz[0]]
        t = qt[4:].copy()
        return Transform(xyzw, t)

    for scene_view_str, result in results.items():
        scene_id, view_id = scene_view_str.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        n_dets = result['rois'].shape[0]
        for n in range(n_dets):
            obj_id = result['rois'][:, 1].astype(np.int)[n]
            label = f'obj_{obj_id:06d}'
            infos.append(dict(
                scene_id=scene_id,
                view_id=view_id,
                score=result['rois'][n, 1],
                label=label,
            ))
            bboxes.append(result['rois'][n, 2:6])
            pose = mat_from_qt(result['poses'][n])
            offset = ycb_offsets[obj_id]
            pose = pose * Transform((0, 0, 0, 1), offset).inverse()
            poses.append(pose.toHomogeneousMatrix())

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)).float(),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
    ).cpu()
    return data

# Load the results from any method that is saved in BOP CSV format

@MEMORY.cache
def load_bop_csv_results(csv_file):
    infos, poses = [], []
    with open(csv_file, 'r') as f:
        lines = f.read().split('\n')[:-1]
    for line in lines:
        scene_id, view_id, obj_id, score, R, t = line.split(',')[:-1]
        scene_id, view_id, obj_id, score = int(scene_id), int(view_id), int(obj_id), float(score)
        label = f'obj_{obj_id:06d}'
        infos.append(dict(
            scene_id=scene_id,
            view_id=view_id,
            score=score,
            label=label,
        ))
        R = np.array([float(rij) for rij in R.split(' ')], dtype=np.float32).reshape(3,3)
        t = 1e-3 * np.array([float(ti) for ti in t.split(' ')], dtype=np.float32).reshape(3)
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3] = t
        poses.append(pose)

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)).float(),
    ).cpu()
    return data


@MEMORY.cache
def load_pix2pose_results(all_detections=True, remove_incorrect_poses=False):
    if all_detections:
        results_path = LOCAL_DATA_DIR / 'saved_detections' / 'tless_pix2pose_retinanet_vivo_all.pkl'
    else:
        results_path = LOCAL_DATA_DIR / 'saved_detections' / 'tless_pix2pose_retinanet_siso_top1.pkl'
    pix2pose_results = pkl.loads(results_path.read_bytes())
    infos, poses, bboxes = [], [], []
    for key, result in pix2pose_results.items():
        scene_id, view_id = key.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        boxes = result['rois']
        scores = result['scores']
        poses_ = result['poses']

        labels = result['labels_txt']
        new_boxes = boxes.copy()
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,3] = boxes[:,2]
        for o, label in enumerate(labels):
            t = poses_[o][:3, -1]
            if remove_incorrect_poses and (np.sum(t) == 0 or np.max(t) > 100):
                pass
            else:
                infos.append(dict(
                    scene_id=scene_id,
                    view_id=view_id,
                    score=scores[o],
                    label=label,
                ))
                bboxes.append(new_boxes[o])
                poses.append(poses_[o])

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
    ).cpu()
    return data


def get_pose_meters(scene_ds):
    ds_name = scene_ds.name

    compute_add = False
    spheres_overlap_check = True
    large_match_threshold_diameter_ratio = 0.5
    if ds_name == 'tless.primesense.test.bop19':
        targets_filename = 'test_targets_bop19.json'
        visib_gt_min = -1
        n_top = -1  # Given by targets
    elif ds_name == 'tless.primesense.test':
        targets_filename = 'all_target_tless.json'
        n_top = 1
        visib_gt_min = 0.1
    elif 'ycbv' in ds_name:
        compute_add = True
        visib_gt_min = -1
        targets_filename = None
        n_top = 1
        spheres_overlap_check = False
    else:
        raise ValueError

    if 'tless' in ds_name:
        object_ds_name = 'tless.eval'
    elif 'ycbv' in ds_name:
        object_ds_name = 'ycbv.bop-compat.eval'  # This is important for definition of symmetric objects
    else:
        raise ValueError

    if targets_filename is not None:
        targets_path = scene_ds.ds_dir / targets_filename
        targets = pd.read_json(targets_path)
        targets = remap_bop_targets(targets)
    else:
        targets = None

    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)

    error_types = ['ADD-S'] + (['ADD(-S)'] if compute_add else [])

    base_kwargs = dict(
        mesh_db=mesh_db,
        exact_meshes=True,
        sample_n_points=None,
        errors_bsz=1,

        # BOP-Like parameters
        n_top=n_top,
        visib_gt_min=visib_gt_min,
        targets=targets,
        spheres_overlap_check=spheres_overlap_check,
    )

    meters = dict()
    for error_type in error_types:
        # For measuring ADD-S AUC on T-LESS and average errors on ycbv/tless.
        meters[f'{error_type}_ntop=BOP_matching=OVERLAP'] = PoseErrorMeter(
            error_type=error_type, consider_all_predictions=False,
            match_threshold=large_match_threshold_diameter_ratio,
            report_error_stats=True, report_error_AUC=True, **base_kwargs)

        if 'ycbv' in ds_name:
            # For fair comparison with PoseCNN/DeepIM on YCB-Video ADD(-S) AUC
            meters[f'{error_type}_ntop=1_matching=CLASS'] = PoseErrorMeter(
                error_type=error_type, consider_all_predictions=False,
                match_threshold=np.inf,
                report_error_stats=False, report_error_AUC=True, **base_kwargs)

        if 'tless' in ds_name:
            meters.update({f'{error_type}_ntop=BOP_matching=BOP':  # For ADD-S<0.1d
                           PoseErrorMeter(error_type=error_type, match_threshold=0.1, **base_kwargs),

                           f'{error_type}_ntop=ALL_matching=BOP':  # For mAP
                           PoseErrorMeter(error_type=error_type, match_threshold=0.1,
                                          consider_all_predictions=True,
                                          report_AP=True, **base_kwargs)})
    return meters


def load_meshes(object_set='tless'):
    if object_set == 'tless':
        object_ds_name, urdf_ds_name = 'tless.bop', 'tless.cad'
    else:
        object_ds_name, urdf_ds_name = 'ycbv.bop-compat.eval', 'ycbv'

    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    return mesh_db

def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.DEBUG)

    logger.info("Starting ...")
    init_distributed_mode()

    parser = argparse.ArgumentParser('BOP csv evaluation')
    parser.add_argument('--csvs', dest='csv_files', default='', type=str, 
            help='Comma-separated list of csv files in BOP format to evaluate '
            + 'for AUC of ADD-S and ADD(-S)')
    parser.add_argument('--config', default='tless-bop', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--job_dir', default='', type=str)
    parser.add_argument('--comment', default='', type=str)
    args = parser.parse_args()

    n_workers = 8
    n_plotters = 8

    n_frames = None
    scene_id = None
    group_id = None
    n_groups = None

    object_set = 'tless'
    if 'tless' in args.config:
        object_set = 'tless'
    elif 'ycbv' in args.config:
        object_set = 'ycbv'
    else:
        raise ValueError(args.config)

    if args.config == 'tless-siso':
        ds_name = 'tless.primesense.test'
        assert n_views == 1
    elif args.config == 'tless-vivo':
        ds_name = 'tless.primesense.test.bop19'
    elif args.config == 'ycbv':
        ds_name = 'ycbv.test.keyframes'
    else:
        raise ValueError(args.config)

    if args.debug:
        if 'tless' in args.config:
            scene_id = None
            group_id = 64
            n_groups = 2
        else:
            scene_id = 48
            n_groups = 2
        n_frames = None
        n_workers = 0
        n_plotters = 0

    n_rand = np.random.randint(1e10)
    save_dir = RESULTS_DIR / f'bop-csv-results-{args.comment}-{n_rand}'
    logger.info(f"SAVE DIR: {save_dir}")

    # Load dataset
    scene_ds = make_scene_dataset(ds_name)

    if scene_id is not None:
        mask = scene_ds.frame_index['scene_id'] == scene_id
        scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)
    if n_frames is not None:
        scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)[:n_frames]

    # Meshes
    mesh_db = load_meshes(object_set=object_set)

    # Evaluation
    predictions_to_evaluate = set()
    all_predictions = dict()
    
    # Load the results from the csvs
    assert args.csv_files != ''
    eval_methods = []
    for csv_file in args.csv_files.split(','):
        # Same method that BOP will call it
        method = csv_file.split('/')[-1].split('_')[0]
        logger.info(f"Loading CSV results {csv_file} for method {method}")
        eval_methods.append(method)
        all_predictions[method] = load_bop_csv_results(csv_file)
        predictions_to_evaluate.add(method)

    if 'ycbv' in ds_name:
        posecnn_detections = load_posecnn_results()
        all_predictions['posecnn'] = posecnn_detections
        predictions_to_evaluate.add('posecnn')
    elif 'tless' in ds_name:
        pix2pose_detections = load_pix2pose_results(all_detections='bop19' in ds_name).cpu()
        all_predictions['pix2pose'] = pix2pose_detections
        predictions_to_evaluate.add('pix2pose')
    else:
        raise ValueError(ds_name)

    all_predictions = OrderedDict({k: v for k, v in sorted(all_predictions.items(), \
            key=lambda item: item[0])})

    # Evaluation.
    meters = get_pose_meters(scene_ds)


    scene_ds_pred = MultiViewWrapper(scene_ds, n_views=1)
    mv_group_ids = list(iter(DistributedSceneSampler(scene_ds, 
            num_replicas=get_world_size(), rank=get_rank())))
    scene_ds_ids = np.concatenate(
            scene_ds_pred.frame_index.loc[mv_group_ids, 'scene_ds_ids'].values)
    sampler = ListSampler(scene_ds_ids)
    eval_runner = PoseEvaluation(scene_ds, meters, n_workers=n_workers,
                                 cache_data=True, batch_size=1, sampler=sampler)

    eval_metrics, eval_dfs = dict(), dict()
    for preds_k, preds in all_predictions.items():
        if preds_k in predictions_to_evaluate:
            logger.info(f"Evaluation : {preds_k} (N={len(preds)})")
            if len(preds) == 0:
                preds = eval_runner.make_empty_predictions()
            eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(preds)
            preds.cpu()
        else:
            logger.info(f"Skipped: {preds_k} (N={len(preds)})")

    all_predictions = gather_predictions(all_predictions)

    metrics_to_print = dict()
    if 'ycbv' in ds_name:
        metrics_to_print.update({
            f'posecnn/ADD(-S)_ntop=1_matching=CLASS/AUC/objects/mean': f'PoseCNN/AUC of ADD(-S)',
            f'posecnn/ADD-S_ntop=1_matching=CLASS/AUC/objects/mean': f'PoseCNN/AUC of ADD-S',
        })
        
        for method in eval_methods:
            metrics_to_print.update({
                f'{method}/ADD(-S)_ntop=1_matching=CLASS/AUC/objects/mean': f'{method}/AUC of ADD(-S)',
                f'{method}/ADD-S_ntop=1_matching=CLASS/AUC/objects/mean': f'{method}/AUC of ADD-S',
            })

    elif 'tless' in ds_name:
        raise NotImplementedError()
        metrics_to_print.update({
            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Singleview/AUC of ADD-S',
            # f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=BOP_matching=BOP/0.1d': f'Singleview/ADD-S<0.1d',
            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=ALL_matching=BOP/mAP': f'Singleview/mAP@ADD-S<0.1d',


            f'{det_key}/ba_output+all_cand/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Multiview (n={args.n_views})/AUC of ADD-S',
            # f'{det_key}/ba_output+all_cand/ADD-S_ntop=BOP_matching=BOP/0.1d': f'Multiview (n={args.n_views})/ADD-S<0.1d',
            f'{det_key}/ba_output+all_cand/ADD-S_ntop=ALL_matching=BOP/mAP': f'Multiview (n={args.n_views}/mAP@ADD-S<0.1d)',
        })
    else:
        raise ValueError

    if get_rank() == 0:
        save_dir.mkdir()
        results = format_results(all_predictions, eval_metrics, eval_dfs, print_metrics=False)
        (save_dir / 'full_summary.txt').write_text(results.get('summary_txt', ''))

        full_summary = results['summary']
        summary_txt = 'Results:'
        for k, v in metrics_to_print.items():
            if k in full_summary:
                summary_txt += f"\n{v}: {full_summary[k]}"
        logger.info(f"{'-'*80}")
        logger.info(summary_txt)
        logger.info(f"{'-'*80}")

        torch.save(results, save_dir / 'results.pth.tar')
        (save_dir / 'summary.txt').write_text(summary_txt)
        logger.info(f"Saved: {save_dir}")


if __name__ == '__main__':
    patch_tqdm()
    main()
    time.sleep(2)
    if get_world_size() > 1:
        torch.distributed.barrier()
