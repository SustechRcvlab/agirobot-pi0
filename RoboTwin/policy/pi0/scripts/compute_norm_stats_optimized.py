"""Optimized version of compute normalization statistics.

This script optimizes the norm stats computation by:
1. Using larger batch sizes
2. Processing entire batches instead of single samples
3. Optional data sampling for faster computation
4. Better memory management
"""

import numpy as np
import tqdm
import tyro
import time

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
        super().__init__()
        self.exclude_dims = set(exclude_dims) if exclude_dims else set()
        self.original_dim = None
    
    def update(self, batch: np.ndarray) -> None:
        """更新统计信息，但跳过指定的维度"""
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        
        num_elements, vector_length = batch.shape
        
        if self.original_dim is None:
            self.original_dim = vector_length
        
        if not self.exclude_dims:
            return super().update(batch)
        
        include_dims = [i for i in range(vector_length) if i not in self.exclude_dims]
        
        if not include_dims:
            return
        
        filtered_batch = batch[:, include_dims]
        super().update(filtered_batch)
    
    def get_statistics(self) -> normalize.NormStats:
        """获取统计信息，为排除的维度填充默认值"""
        if not self.exclude_dims or self.original_dim is None:
            return super().get_statistics()
        
        filtered_stats = super().get_statistics()
        total_dims = self.original_dim
        
        full_mean = np.zeros(total_dims)
        full_std = np.ones(total_dims)
        full_q01 = np.zeros(total_dims) if filtered_stats.q01 is not None else None
        full_q99 = np.ones(total_dims) if filtered_stats.q99 is not None else None
        
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
    
    if data_config.replace_gripper_in_state:
        gripper_replacement = ReplaceGripperInState(
            gripper_dims=list(data_config.gripper_dimensions),
            use_action_values=True
        )
        transforms.append(gripper_replacement)
        print(f"Added gripper replacement for norm stats computation, dimensions: {data_config.gripper_dimensions}")
    
    transforms.append(RemoveStrings())
    
    dataset = _data_loader.TransformedDataset(dataset, transforms)
    return data_config, dataset


def main(
    config_name: str, 
    max_frames: int | None = None,
    sample_ratio: float = 1.0,  # 添加采样比例参数
    batch_size: int = 64,       # 增加默认批次大小
    num_workers: int = 16       # 增加工作进程数
):
    """
    Args:
        config_name: 配置名称
        max_frames: 最大处理帧数
        sample_ratio: 采样比例 (0.0-1.0)，用于快速测试
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
    """
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    print(f"数据集总大小: {num_frames:,} 个样本")
    
    # 应用采样比例
    if sample_ratio < 1.0:
        num_frames = int(num_frames * sample_ratio)
        print(f"采样后大小: {num_frames:,} 个样本 (采样比例: {sample_ratio:.1%})")
    
    shuffle = False
    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True
        print(f"限制处理数量: {num_frames:,} 个样本")

    # 优化的数据加载器配置
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,  # 增加批次大小
        num_workers=num_workers,      # 增加工作进程
        shuffle=shuffle,
        num_batches=num_frames,
    )
    
    def get_val(d, key):
        for k in key.split('.'):
            try:
                d = d[k] if isinstance(d, dict) else d[int(k)]
            except (KeyError, IndexError, ValueError, TypeError) as e:
                raise KeyError(f"获取 '{key}' 失败，在层级 '{k}' 出错：{e}")
        return np.asarray(d)
    
    def preprocess_batch(batch):
        """优化的批处理预处理函数"""
        try:
            state = get_val(batch, "state")
            actions = get_val(batch, "actions")
            return {
                "state": state,
                "actions": actions,
            }
        except Exception as e:
            print(f"预处理批次时出错: {e}")
            print("Batch keys:", batch.keys())
            raise
    
    keys = ["state", "actions"]
    gripper_dims = [7, 15]
    
    # 创建统计对象
    stats = {}
    for key in keys:
        if key in ["state", "actions"]:
            stats[key] = SelectiveRunningStats(exclude_dims=gripper_dims)
            print(f"为 '{key}' 创建统计对象，跳过夹爪维度: {gripper_dims}")
        else:
            stats[key] = normalize.RunningStats()
    
    print(f"开始计算归一化统计信息...")
    print(f"配置: batch_size={batch_size}, num_workers={num_workers}")
    
    start_time = time.time()
    processed_samples = 0  # 这里记录的是实际处理的样本数（不是数据点数）
    total_data_points = 0  # 这里记录的是统计计算用的数据点数
    
    # 计算总批次数
    total_batches = (num_frames + batch_size - 1) // batch_size
    
    try:
        for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, total=total_batches, desc="Computing stats")):
            # 预处理整个批次
            batch_processed = preprocess_batch(batch)
            batch.update(batch_processed)
            
            # 记录当前批次的实际样本数
            current_batch_size = len(batch[keys[0]])  # 使用第一个key获取批次大小
            processed_samples += current_batch_size
            
            for key in keys:
                values = np.asarray(batch[key])  # 处理整个批次，不只是第一个样本
                
                # 打印维度信息（仅第一个batch）
                if batch_idx == 0:
                    print(f"'{key}' 批次维度: {values.shape}, 数据类型: {values.dtype}")
                    if key in ["state", "actions"]:
                        print(f"  -> 将跳过夹爪维度: {gripper_dims}")
                
                # 重塑为 (batch_size * sequence_length, feature_dim)
                if values.ndim == 3:  # (batch_size, sequence_length, feature_dim)
                    values_reshaped = values.reshape(-1, values.shape[-1])
                    # 只在第一个key时更新数据点计数，避免重复计算
                    if key == keys[0]:
                        total_data_points += values_reshaped.shape[0]
                elif values.ndim == 2:  # (batch_size, feature_dim)
                    values_reshaped = values
                    # 只在第一个key时更新数据点计数，避免重复计算
                    if key == keys[0]:
                        total_data_points += values_reshaped.shape[0]
                else:
                    print(f"警告: '{key}' 的维度不符合预期: {values.shape}")
                    continue
                
                # 更新统计信息
                stats[key].update(values_reshaped)
            
            # 每100个批次打印进度
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                sample_rate = processed_samples / elapsed
                data_point_rate = total_data_points / elapsed
                print(f"已处理 {processed_samples:,}/{num_frames:,} 样本 ({total_data_points:,} 数据点), 样本速度: {sample_rate:.1f} samples/s, 数据点速度: {data_point_rate:.1f} points/s")
    
    except KeyboardInterrupt:
        print("\n计算被中断，保存当前统计信息...")
    except Exception as e:
        print(f"\n计算过程中出错: {e}")
        raise
    
    # 计算最终统计信息
    norm_stats = {}
    for key, stat in stats.items():
        if stat._count > 0:  # 确保有数据
            stat_result = stat.get_statistics()
            norm_stats[key] = stat_result
            print(f"'{key}' 统计信息:")
            print(f"  样本数: {stat._count:,}")
            print(f"  均值形状: {stat_result.mean.shape}")
            print(f"  标准差形状: {stat_result.std.shape}")
            print(f"  均值范围: [{stat_result.mean.min():.4f}, {stat_result.mean.max():.4f}]")
            print(f"  标准差范围: [{stat_result.std.min():.4f}, {stat_result.std.max():.4f}]")
            if key in ["state", "actions"]:
                print(f"  夹爪维度 {gripper_dims} 已跳过归一化计算")
        else:
            print(f"警告: '{key}' 没有收集到数据")

    if norm_stats:
        output_path = config.assets_dirs / data_config.repo_id
        print(f"保存统计信息到: {output_path}")
        normalize.save(output_path, norm_stats)
        
        elapsed = time.time() - start_time
        print(f"总计算时间: {elapsed/60:.1f} 分钟")
        print(f"实际处理样本数: {processed_samples:,}")
        print(f"统计计算数据点数: {total_data_points:,}")
        print(f"平均样本处理速度: {processed_samples/elapsed:.1f} samples/s")
        print(f"平均数据点处理速度: {total_data_points/elapsed:.1f} points/s")
    else:
        print("错误: 没有收集到任何统计信息！")


if __name__ == "__main__":
    tyro.cli(main)
