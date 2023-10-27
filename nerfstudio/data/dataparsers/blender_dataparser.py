# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Literal
from PIL import Image
import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType, CAMERA_MODEL_TO_TYPE
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.cameras import camera_utils
from nerfstudio.utils.rich_utils import CONSOLE


def get_foreground_masks(image_idx: int, fg_masks):
    """function to process additional foreground_masks

    Args:
        image_idx: specific image index to work with
        fg_masks: foreground_masks
    """

    # sensor depth
    fg_mask = fg_masks[image_idx]

    return {"fg_mask": fg_mask}

def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []

@dataclass
class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Blender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""

@dataclass
class Blender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlenderDataParserConfig

    def __init__(self, config: BlenderDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        indices = list(range(len(meta["frames"])))
        image_filenames = []
        poses = []
        foreground_mask_images = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", ""))

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )
            if self.config.include_foreground_mask:
                # load foreground mask
                foreground_mask = np.array(Image.open(self.config.data / frame["masks"]), dtype="uint8")
                foreground_mask = foreground_mask[..., :1]
                foreground_mask_images.append(torch.from_numpy(foreground_mask).float() / 255.0)


            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            print(foreground_mask_images)
        # poses = np.array(poses).astype(np.float32)

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_poses=self.config.center_poses,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= self.config.scale_factor

        # img_0 = imageio.v2.imread(image_filenames[0])
        # image_height, image_width = img_0.shape[:2]
        # camera_angle_x = float(meta["camera_angle_x"])
        # focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        # cx = image_width / 2.0
        # # cy = image_height / 2.0
        # camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
        #
        # # in x,y,z order
        # camera_to_world[..., 3] *= self.scale_factor
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_type=camera_type,
        )

        additional_inputs_dict = {}
        if self.config.include_foreground_mask:
            additional_inputs_dict["foreground_masks"] = {
                "func": get_foreground_masks,
                "kwargs": {"fg_masks": filter_list(foreground_mask_images, indices)},
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            # dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs