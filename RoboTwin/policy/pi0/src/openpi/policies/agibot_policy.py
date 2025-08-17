import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_agibot_example() -> dict:
    """Creates a random input example for the AgiBot policy."""
    return {
        "observation": {
            "states": np.ones((16,)),
            "images": {
                "head": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
                "hand_left": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
                "hand_right": np.random.randint(256, size=(480, 848, 3), dtype=np.uint8),
            }
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AgiBotInputs(transforms.DataTransformFn):
    """Inputs for the AgiBot policy.

    Expected inputs:
    - observation.images: dict[name, img] where img is [height, width, channel]. name must be in EXPECTED_CAMERAS.
    - observation.states: [16]
    - actions: [action_horizon, 16]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard AgiBot space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "head",
        "hand_left", 
        "hand_right",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_agibot(data, adapt_to_pi=self.adapt_to_pi)

        # Get the state. We are padding from 16 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that head image always exists as base image.
        base_image = in_images["head"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "hand_left",
            "right_wrist_0_rgb": "hand_right",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgiBotOutputs(transforms.DataTransformFn):
    """Outputs for the AgiBot policy."""

    # If true, this will convert the joint and gripper values from the standard AgiBot space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 dims.
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between agibot and pi joint angles."""
    # AgiBot has 16 dimensions: 7 joints + 1 gripper for each arm
    # return np.array([1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1])
    return np.ones(16)


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Similar to Aloha but adapted for AgiBot gripper characteristics
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by AgiBot.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are adapted for AgiBot
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_agibot(data: dict, *, adapt_to_pi: bool = False) -> dict:
    """Decode AgiBot dataset sample into standard format."""
    # print("Data keys:", list(data.keys()))

    # 解码 state
    state = data.get("state", None)

    def convert_image(img):
        """Ensure image is (H, W, C) and uint8."""
        img = np.asarray(img)
        # 如果是 (C, H, W) → 转为 (H, W, C)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        # 如果是 float → 转成 uint8
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0, 1)  # 防止超出范围
            img = (img * 255).astype(np.uint8)
        # 如果是灰度 → 扩展成 3 通道
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img

    # 处理 images
    images = data.get("images", {})
    images_dict = {name: convert_image(img) for name, img in images.items()}

    # 构造返回结构
    result = {
        "images": images_dict,
        "state": state
    }

    # 可选字段
    if "actions" in data:
        result["actions"] = data["actions"]

    # if "prompt" in data:
    result["prompt"] = " "

    return result



def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the AgiBot runtime.
        # Gripper indices are 7 (left) and 15 (right) for 16-dim state
        state[[7, 15]] = _gripper_to_angular(state[[7, 15]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        # Gripper indices are 7 (left) and 15 (right) for 16-dim actions
        actions[:, [7, 15]] = _gripper_from_angular(actions[:, [7, 15]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        # Gripper indices are 7 (left) and 15 (right) for 16-dim actions
        actions[:, [7, 15]] = _gripper_from_angular_inv(actions[:, [7, 15]])
    return actions