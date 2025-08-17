from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
# 我们定义的结构和原来的不一样 所以需要重新声明一个从lerobot_dataset出发的基类
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from pathlib import Path

class AgiBotDataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = False,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        image_path: str | None = None
    ) -> "LeRobotDataset":
        """Create a agibotworld LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        # obj.image_path = Path(image_path) if image_path else None
        obj.image_path="/dataset/lerobot_dataset"

        return obj

    def __getitem__(self, idx: int) -> dict:
        """Override the parent __getitem__ to include image loading from file paths."""
        # Get the base item from parent class
        item = super().__getitem__(idx)
        
        # Load images from file paths and add to item
        image_data = self._query_images(None, idx)
        # image_data = {}
        # 在第75行，设置适当的空张量
        # image_data = {
        #     "observation.images.head": torch.zeros((3, 224, 224)),  # 根据实际尺寸调整
        #     "observation.images.hand_left": torch.zeros((3, 224, 224)),
        #     "observation.images.hand_right": torch.zeros((3, 224, 224))
        # }

        # Map the loaded images to the expected format
        if image_data:
            # Map camera names to expected keys
            camera_mapping = {
                "observation.images.head": "head",
                "observation.images.hand_left": "hand_left", 
                "observation.images.hand_right": "hand_right"
            }
            
            for camera_key, image_tensor in image_data.items():
                if camera_key in camera_mapping:
                    expected_key = camera_mapping[camera_key]
                    item[f"observation.images.{expected_key}"] = image_tensor
            # print(f"observation images keys: {item.keys()}")
        # reset some 
        return item

    def _query_images(self, query_indices: dict[str, list[int]] | None, idx: int) -> dict[str, torch.Tensor]:
        """Load images directly from file paths constructed from observation.images.path data."""
        item = {}

        # Check if we have observation.images.path in the dataset
        if "observation.images.path" not in self.hf_dataset.column_names:
            return item

        if query_indices is not None:
            # Load multiple images for delta timestamps
            if "observation.images.path" in query_indices:
                selected_data = self.hf_dataset.select(query_indices["observation.images.path"])
                path_arrays = selected_data["observation.images.path"]

                all_images = {}
                for camera_name in ["hand_left_color", "hand_right_color", "head_color"]:
                    images = []
                    for path_array in path_arrays:
                        img_path = self._construct_image_path(path_array, camera_name)
                        image = self._load_single_image(img_path)
                        images.append(image)
                    all_images[f"observation.images.{camera_name}"] = torch.stack(images)
                item.update(all_images)
        else:
            # Load single image for current timestamp
            path_array = self.hf_dataset[idx]["observation.images.path"]

            # Load all camera images for current timestamp
            for camera_name in ["hand_left_color", "hand_right_color", "head_color"]:
                img_path = self._construct_image_path(path_array, camera_name)
                key_name = camera_name.replace("_color", "")
                item[f"observation.images.{key_name}"] = self._load_single_image(img_path)
                # print(f"Add item observation.images.{key_name}")
                # print("Keys:",item.keys())
        # item['prompt'] = " "
        return item

    def _construct_image_path(self, path_array: list | torch.Tensor, camera_name: str) -> Path:
        """Construct image path from path array and camera name."""
        if isinstance(path_array, torch.Tensor):
            path_array = path_array.tolist()

        task_id, job_id, episode_id, frame_index = path_array
        image_path = Path(f"/dataset/SimDatas/{task_id}/{job_id}/A2D0015AB00061/{episode_id}/camera/{frame_index}/{camera_name}.jpg")

        return image_path

    def _load_single_image(self, img_path: Path) -> torch.Tensor:
        """Load a single image and convert to torch tensor."""
        import PIL.Image

        # Check if file exists
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Load image
        image = PIL.Image.open(img_path).convert('RGB')

        # Convert to numpy array and then to tensor [C, H, W] with values in [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        return image_tensor

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Override to exclude image path from normal querying."""
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys and key != "observation.images.path"
        }

T_co = TypeVar("T_co", covariant=True)

# class AgiLerobotDataset(lerobot_dataset.LeRobotDataset):
class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id="",root=repo_id)
    print(f"meta_data={dataset_meta}")
    dataset = AgiBotDataset(
        repo_id="",
        root=data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    # if data_config.prompt_from_task:
    #     dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    from openpi.transforms_gripper_replacement import ReplaceGripperInStatePostNorm
    
    norm_stats = {}
    #print("正在转换数据集")
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found. "
                             "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.")
        norm_stats = data_config.norm_stats

    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
    ]
    
    # Add gripper replacement transform after normalization if enabled
    if data_config.replace_gripper_in_state:
        gripper_replacement = ReplaceGripperInStatePostNorm(
            gripper_dims=list(data_config.gripper_dimensions),
            use_action_values=True
        )
        transforms.append(gripper_replacement)
        print(f"Added gripper replacement transform for dimensions: {data_config.gripper_dimensions}")
    
    transforms.extend(data_config.model_transforms.inputs)

    return TransformedDataset(dataset, transforms)


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):

        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B", )),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            # print(list(batch.keys()))
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
