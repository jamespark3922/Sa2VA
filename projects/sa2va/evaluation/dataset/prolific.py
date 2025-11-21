import os
import json

import mmengine
from pathlib import Path    
from glob import glob

from PIL import Image
import copy

from mmengine.dist import master_only

from .base_eval_dataset import BaseEvalDataset

from datasets import load_from_disk

SEG_PROMPT = "<image>\nPlease segment {}."
SEG_PROMPT_QUESTION = "<image>\n {} Please respond with a segmentation mask."


class ProlificDataset(BaseEvalDataset):
    def __init__(self,
                 image_folder,
                 hf_meta_path,
                 use_disk_frames=True):
        """
        HF-based Prolific evaluation dataset.

        Args:
            image_folder (str): root directory containing video frame folders:
                                image_folder/<video_id>/*.jpg|*.png
            hf_meta_path (str): path to HuggingFace dataset on disk (load_from_disk)
                                with at least 'video' and 'exp' columns.
            use_disk_frames (bool): if True, derive frame list from disk when
                                    'frames' is not present in the dataset.
        """
        super().__init__()

        self.image_folder = image_folder
        self.hf_meta_path = hf_meta_path
        self.use_disk_frames = use_disk_frames

        # Build metas and vid2metaid from HF dataset
        vid2metaid, metas = self.hf_preprocess(hf_meta_path)
        self.vid2metaid = vid2metaid
        self.videos = list(self.vid2metaid.keys())
        self.text_data = metas   # list of meta dicts, same as before

    def __len__(self):
        return len(self.text_data)

    def real_len(self):
        return len(self.text_data)

    def hf_preprocess(self, hf_meta_path):
        """
        Replace json_file_preprocess: read from HF dataset instead of JSON.

        Expected HF fields per row:
          - 'video': video id (str)
          - 'exp': expression text (str)
          - optional 'frames': list[str] of frame name stems
          - optional 'exp_id': expression id
          - optional 'id': something like '{video}_{exp_id}'

        Returns:
          vid2metaid: dict[video_id] -> list of meta indices
          metas: list of meta dicts with keys:
                 'video', 'exp', 'frames', 'exp_id', 'length'
        """
        ds = load_from_disk(hf_meta_path)

        metas = []
        vid2metaid = {}

        for i in range(len(ds)):
            row = ds[i]

            # --- required fields ---
            video = row["video"]
            exp = row["exp"]

            # --- frames: from dataset or from disk ---
            if "frames" in ds.column_names:
                frames = row["frames"]  # list of frame stems
            else:
                # derive from disk
                video_dir = Path(self.image_folder) / video
                if not video_dir.exists():
                    raise FileNotFoundError(f"Video directory not found: {video_dir}")

                frame_files = sorted(
                    glob(os.path.join(str(video_dir), "*.jpg"))
                    + glob(os.path.join(str(video_dir), "*.png"))
                )
                if len(frame_files) == 0:
                    raise RuntimeError(f"No frames found for video {video} in {video_dir}")
                frames = [Path(f).stem for f in frame_files]

            vid_len = len(frames)

            qid = row["id"]
            # robust: only split once from the right
            _, exp_id = qid.rsplit("_", 1)

            meta = {
                "video": video,
                "exp": exp,
                "frames": frames,
                "id": qid,
                "exp_id": exp_id,
                "length": vid_len,
                "w": row["w"],
                "h": row["h"],
            }
            metas.append(meta)

            if video not in vid2metaid:
                vid2metaid[video] = []
            vid2metaid[video].append(len(metas) - 1)

        return vid2metaid, metas

    def __getitem__(self, index):
        video_obj_info = copy.deepcopy(self.text_data[index])
        exp = video_obj_info["exp"]

        data_dict = {}

        video_id = video_obj_info["video"]
        frame_stems = video_obj_info["frames"]

        frame_paths = [
            os.path.join(self.image_folder, video_id, stem + ".jpg")
            for stem in frame_stems
        ]
        # fallback for .png if .jpg doesn't exist
        frame_paths = [
            p if os.path.exists(p) else p.replace(".jpg", ".png")
            for p in frame_paths
        ]

        # Load all frames from the video directory
        # video_dir = os.path.join(self.image_folder, video_id)
        # frame_paths = sorted([
        #     os.path.join(video_dir, f) 
        #     for f in os.listdir(video_dir)
        #     if f.endswith(('.jpg', '.png', '.jpeg'))
        # ])

        images = []
        ori_width, ori_height = None, None
        for frame_path in frame_paths:
            frame_image = Image.open(frame_path).convert("RGB")
            if ori_height is None:
                ori_width, ori_height = frame_image.size
            else:
                assert ori_width == frame_image.size[0]
                assert ori_height == frame_image.size[1]
            images.append(frame_image)

        data_dict["type"] = "video"
        data_dict["index"] = index
        data_dict["video_id"] = video_id
        data_dict["images"] = images
        data_dict["frames"] = frame_stems
        data_dict["exp_id"] = video_obj_info["exp_id"]
        data_dict["id"] = video_obj_info["id"]

        # data_dict["frames"] = frame_stems
        data_dict["text_prompt"] = (
            SEG_PROMPT.format(exp)
            if "?" not in exp
            else SEG_PROMPT_QUESTION.format(exp)
        )
        data_dict["image_folder"] = self.image_folder

        data_dict["length"] = video_obj_info["length"]
        data_dict["ori_height"] = ori_height
        data_dict["ori_width"] = ori_width

        return data_dict
