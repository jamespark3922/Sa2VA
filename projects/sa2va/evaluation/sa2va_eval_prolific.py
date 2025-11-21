import argparse
import json
import os

import mmengine
import numpy as np
from PIL import Image

import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from projects.sa2va.evaluation.dataset import ProlificDataset
from projects.sa2va.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu

import concurrent.futures
from pycocotools import mask as cocomask


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def mask_save(item, mask_prediction, work_dir):
    # vid_id = item['video_id']
    # exp_id = item['exp_id']
    id = item['id']
    save_path = os.path.join(work_dir, 'Annotations', id)
    mmengine.mkdir_or_exist(save_path)
    print(f"Saving masks to {save_path}")
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert('L')
        file_name = item['frames'][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)


DATASETS_INFO = {
    'MEVIS': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/mevis/MeViS_release/valid_u',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/mevis/MeViS_release/valid_u/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/mevis/MeViS_release/valid_u/annotation/largest_center',
    },
    'MEVIS-VALID': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/mevis/MeViS_release/valid',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/mevis/MeViS_release/valid/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/mevis/MeViS_release/valid/annotation/largest_center',
    },
    'REF-YT-VOS': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/Ref-YT-VOS/valid',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/Ref-YT-VOS/valid/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/Ref-YT-VOS/valid/annotation/largest_center',
    },
    'REF-DAVIS17': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/Ref-DAVIS17/valid',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/Ref-DAVIS17/valid/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/Ref-DAVIS17/valid/annotation/largest_center',
    },
    'REASON_VOS': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/ReasonVOS/',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/ReasonVOS/JPEGImages/',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/ReasonVOS/annotation/largest_center',
    },
    'PROLIFIC': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val/annotation/largest_center',
    },
    'PROLIFIC_GENERAL': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_general',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_general/annotation/largest_center',
    },
    'PROLIFIC_ANIMALS': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_animals',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_animals/annotation/largest_center',
    },
    'PROLIFIC_DANCE': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_dance',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_dance/annotation/largest_center',
    },
    'PROLIFIC_PEDESTRIAN': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_pedestrian',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_pedestrian/annotation/largest_center',    
    },
    'PROLIFIC_SPORTS': {
        'data_root': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_sports',
        'image_folder': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val/JPEGImages',
        'expression_file': '/weka/oe-training-default/mm-olmo/video_datasets/prolific/video_text_queries_filtered_111025_val_category_sports/annotation/largest_center',    
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='RefVOS')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='MEVIS',
        help='Specify a dataset')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--deepspeed', type=str, default=None) # dummy
    parser.add_argument('--data_root', default='./data', help='Root directory for all datasets.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()

    # Update dataset paths with data_root
    # for key, info in DATASETS_INFO.items():
    #     for path_key, path_val in info.items():
    #         if path_val is not None and ('folder' in path_key or 'file' in path_key or 'root' in path_key):
    #             DATASETS_INFO[key][path_key] = os.path.join(args.data_root, os.path.relpath(path_val, './data'))

    dataset_info = DATASETS_INFO[args.dataset]


    dataset = ProlificDataset(
        image_folder=dataset_info['image_folder'],
        hf_meta_path=dataset_info['expression_file'],
    )

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = 'work_dirs/foobar'

    if args.launcher == 'none':
        rank = 0
        world_size = 1
    elif args.launcher == 'pytorch':
        import datetime
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
    elif args.launcher == 'slurm':
        _init_dist_slurm('nccl')
        rank, world_size = get_dist_info()

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    is_qwen = 'qwen' in args.model_path.lower()

    if is_qwen:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        processor = None

    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=8,
        pin_memory=False,
        collate_fn=lambda x:x[0],
    )
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    for item in tqdm.tqdm(dataloader):
        with torch.no_grad():
            print('logging', item['id'], len(item['images']))
            if is_qwen:
                result = model.predict_forward(
                    video=item['images'][:950],
                    text=item['text_prompt'],
                    tokenizer=tokenizer,
                    processor=processor,
                )
            else:
                result = model.predict_forward(
                    video=item['images'][:950],
                    text=item['text_prompt'],
                    tokenizer=tokenizer,
                )

        text_idx = 0
        text_prediction = result['prediction']
        if len(result['prediction_masks']) > 0:
            mask_prediction = result['prediction_masks'][text_idx]
        else:
            #print(text_prediction)
            mask_prediction = np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)

        if args.submit:
            async_func(executor, mask_save, item=item, mask_prediction=mask_prediction, work_dir=work_dir)
            encoded_mask = None
        else:
            encoded_mask = mask_to_rle(mask_prediction)

        result = {
            'index': item['index'],
            'video_id': item['video_id'],
            'exp_id': item['exp_id'],
            'text_prediction': text_prediction,
            'frames': item['frames'],
            'exp': item['text_prompt'],
            'prediction_masks': encoded_mask,

        }
        results.append(result)


    executor.shutdown(wait=True)
    print(f'[Rank {rank}] : Finished.')
    
    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item['video_id']
                exp_id = item['exp_id']
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            work_dir = os.path.join(work_dir, args.dataset)
            os.makedirs(work_dir, exist_ok=True)
            json.dump(final_results, open(f'{work_dir}/results.json', 'w'))

    if rank == 0:
        print('Done')
