"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class SelectiveRunningStats(normalize.RunningStats):
    """运行统计类，可以跳过指定的维度（如夹爪维度）"""
    
    def __init__(self, exclude_dims=None):
        """
        Args:
            exclude_dims: 要排除的维度列表，如 [0, 17] 表示跳过第0和第17维
        """
        super().__init__()
        self.exclude_dims = set(exclude_dims) if exclude_dims else set()
        self.original_dim = None  # 记录原始维度
    
    def update(self, batch: np.ndarray) -> None:
        """更新统计信息，但跳过指定的维度"""
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        
        num_elements, vector_length = batch.shape
        
        # 记录原始维度
        if self.original_dim is None:
            self.original_dim = vector_length
        
        # 如果没有要排除的维度，使用原始方法
        if not self.exclude_dims:
            return super().update(batch)
        
        # 创建掩码，排除指定维度
        include_dims = [i for i in range(vector_length) if i not in self.exclude_dims]
        
        if not include_dims:  # 如果所有维度都被排除，直接返回
            return
        
        # 只处理非排除的维度
        filtered_batch = batch[:, include_dims]
        super().update(filtered_batch)
    
    def get_statistics(self) -> normalize.NormStats:
        """获取统计信息，为排除的维度填充默认值"""
        if not self.exclude_dims or self.original_dim is None:
            return super().get_statistics()
        
        # 获取非排除维度的统计信息
        filtered_stats = super().get_statistics()
        
        # 重建完整的统计信息，为排除的维度设置默认值
        total_dims = self.original_dim
        
        full_mean = np.zeros(total_dims)
        full_std = np.ones(total_dims)  # 标准差设为1，避免除零
        full_q01 = np.zeros(total_dims) if filtered_stats.q01 is not None else None
        full_q99 = np.ones(total_dims) if filtered_stats.q99 is not None else None
        
        # 填充非排除维度的值
        include_dims = [i for i in range(total_dims) if i not in self.exclude_dims]
        if include_dims and len(filtered_stats.mean) > 0:
            full_mean[include_dims] = filtered_stats.mean
            full_std[include_dims] = filtered_stats.std
            if full_q01 is not None:
                full_q01[include_dims] = filtered_stats.q01
            if full_q99 is not None:
                full_q99[include_dims] = filtered_stats.q99
        
        return normalize.NormStats(
            mean=full_mean,
            std=full_std,
            q01=full_q01,
            q99=full_q99
        )



def create_dataset(config: _config.TrainConfig, ) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    from openpi.transforms_gripper_replacement import ReplaceGripperInState
    
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    
    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ]
    
    # Add gripper replacement transform before normalization for stats computation if enabled
    if data_config.replace_gripper_in_state:
        gripper_replacement = ReplaceGripperInState(
            gripper_dims=list(data_config.gripper_dimensions),
            use_action_values=True
        )
        transforms.append(gripper_replacement)
        print(f"Added gripper replacement for norm stats computation, dimensions: {data_config.gripper_dimensions}")
    
    # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
    transforms.append(RemoveStrings())
    
    dataset = _data_loader.TransformedDataset(dataset, transforms)
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=8,
        num_workers=8,
        shuffle=False,
        num_batches=num_frames,
    )
    def get_val(d, key):
        for k in key.split('.'):
            try:
                d = d[k] if isinstance(d, dict) else d[int(k)]
            except (KeyError, IndexError, ValueError, TypeError) as e:
                raise KeyError(f"获取 '{key}' 失败，在层级 '{k}' 出错：{e}")
        return np.asarray(d)
    
    def preprocess(batch):
        # 输出batch的所有键
        print("Batch keys:", batch.keys())
        # states=np.asarray(batch["state"])
        # state_joint_positions = np.asarray(batch['observation.states.joint.position'])
        state=get_val(batch, "state")
        actions=get_val(batch,"actions")
        return {
            "state": state,
            "actions": actions,
        }
    keys = ["state", "actions"]
    # 定义夹爪维度（第0和第17维）
    # 0 1 2 3 4 5 6 7（夹爪） 8 9 10 11 12 13 14 15 
    gripper_dims = [7, 15]
    
    # 为每个键创建统计对象，跳过夹爪维度
    stats = {}
    for key in keys:
        if key in ["state", "actions"]:  # 对于state和actions，跳过夹爪维度
            stats[key] = SelectiveRunningStats(exclude_dims=gripper_dims)
            print(f"为 '{key}' 创建统计对象，跳过夹爪维度: {gripper_dims}")
        else:
            stats[key] = normalize.RunningStats()
    
    print(f"Computing normalization statistics for {num_frames} frames...")
    # tqdm 用于进度条显示
    for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
        # 输出一些batch的信息（比如他有哪些）
        batch_add= preprocess(batch)
        # 在batch中添加新的键值对batch_add
        batch.update(batch_add)
        for key in keys:
            values = np.asarray(batch[key][0])
            # 打印维度信息（仅第一个batch）
            if stats[key]._count == 0:
                print(f"'{key}' 的维度: {values.shape}, 数据类型: {values.dtype}")
                if key in ["state", "actions"]:
                    print(f"  -> 将跳过夹爪维度: {gripper_dims}")
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {}
    for key, stat in stats.items():
        stat_result = stat.get_statistics()
        norm_stats[key] = stat_result
        print(f"'{key}' 统计信息:")
        print(f"  均值形状: {stat_result.mean.shape}")
        print(f"  标准差形状: {stat_result.std.shape}")
        if key in ["state", "actions"]:
            print(f"  夹爪维度 {gripper_dims} 已跳过归一化计算")

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
