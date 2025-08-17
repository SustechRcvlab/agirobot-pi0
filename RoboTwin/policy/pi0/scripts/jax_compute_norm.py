"""JAX-accelerated version of compute normalization statistics.

This script uses JAX for significantly faster norm stats computation by:
1. JIT compilation for optimized performance
2. Vectorized operations across all features
3. GPU acceleration (if available)
4. Memory-efficient streaming computation
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import tqdm
import tyro
import time
from typing import Dict, Any

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

# 启用JAX 64位精度
jax.config.update("jax_enable_x64", True)

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

@jit
def compute_batch_stats_jax(data: jnp.ndarray, include_indices: jnp.ndarray = None):
    """JIT编译的批次统计计算"""
    # data shape: (batch_size, seq_len, features) or (batch_size * seq_len, features)
    if data.ndim == 3:
        data_flat = data.reshape(-1, data.shape[-1])
    else:
        data_flat = data
    
    # 应用包含索引（用于跳过夹爪维度）
    if include_indices is not None:
        data_flat = data_flat[:, include_indices]
    
    # 计算统计量
    batch_sum = jnp.sum(data_flat, axis=0, dtype=jnp.float64)
    batch_sum_sq = jnp.sum(data_flat ** 2, axis=0, dtype=jnp.float64)
    batch_count = jnp.array(data_flat.shape[0], dtype=jnp.int64)
    
    return batch_sum, batch_sum_sq, batch_count

@jit 
def update_running_stats_jax(running_sum, running_sum_sq, running_count, 
                            batch_sum, batch_sum_sq, batch_count):
    """JIT编译的累积统计更新"""
    new_sum = running_sum + batch_sum
    new_sum_sq = running_sum_sq + batch_sum_sq
    new_count = running_count + batch_count
    return new_sum, new_sum_sq, new_count

@jit
def finalize_stats_jax(total_sum, total_sum_sq, total_count):
    """JIT编译的最终统计计算"""
    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean ** 2)
    variance = jnp.maximum(variance, 1e-8)
    std = jnp.sqrt(variance)
    return mean, std

class JAXRunningStats:
    """JAX版本的运行统计类"""
    
    def __init__(self, exclude_dims=None):
        self.exclude_dims = set(exclude_dims) if exclude_dims else set()
        self.original_dim = None
        self.include_indices = None
        self.running_sum = None
        self.running_sum_sq = None
        self.running_count = jnp.array(0, dtype=jnp.int64)
        
    def _initialize(self, feature_dim):
        """初始化JAX数组"""
        self.original_dim = feature_dim
        
        if self.exclude_dims:
            # 创建包含索引（整数数组而不是布尔数组）
            include_list = [i for i in range(feature_dim) if i not in self.exclude_dims]
            self.include_indices = jnp.array(include_list, dtype=jnp.int32)
            effective_dim = len(include_list)
        else:
            self.include_indices = None
            effective_dim = feature_dim
            
        self.running_sum = jnp.zeros(effective_dim, dtype=jnp.float64)
        self.running_sum_sq = jnp.zeros(effective_dim, dtype=jnp.float64)
        
    def update(self, batch: np.ndarray):
        """更新统计信息"""
        # 转换为JAX数组
        data_jax = jnp.array(batch, dtype=jnp.float32)
        
        # 初始化（如果需要）
        if self.running_sum is None:
            if data_jax.ndim == 3:
                feature_dim = data_jax.shape[-1]
            else:
                feature_dim = data_jax.shape[-1]
            self._initialize(feature_dim)
        
        # 计算批次统计
        batch_sum, batch_sum_sq, batch_count = compute_batch_stats_jax(
            data_jax, self.include_indices
        )
        
        # 更新累积统计
        self.running_sum, self.running_sum_sq, self.running_count = update_running_stats_jax(
            self.running_sum, self.running_sum_sq, self.running_count,
            batch_sum, batch_sum_sq, batch_count
        )
    
    def get_statistics(self) -> normalize.NormStats:
        """获取最终统计信息"""
        if self.running_sum is None or self.running_count == 0:
            raise ValueError("没有数据用于计算统计信息")
        
        # 计算最终统计
        mean, std = finalize_stats_jax(
            self.running_sum, self.running_sum_sq, self.running_count
        )
        
        # 转换回NumPy
        mean_np = np.array(mean, dtype=np.float32)
        std_np = np.array(std, dtype=np.float32)
        
        # 如果有排除的维度，需要填充完整数组
        if self.exclude_dims and self.original_dim:
            full_mean = np.zeros(self.original_dim, dtype=np.float32)
            full_std = np.ones(self.original_dim, dtype=np.float32)
            
            include_indices = [i for i in range(self.original_dim) if i not in self.exclude_dims]
            full_mean[include_indices] = mean_np
            full_std[include_indices] = std_np
            
            return normalize.NormStats(
                mean=full_mean,
                std=full_std,
                q01=None,  # JAX版本暂不计算分位数
                q99=None
            )
        else:
            return normalize.NormStats(
                mean=mean_np,
                std=std_np,
                q01=None,
                q99=None
            )

def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
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
    sample_ratio: float = 1.0,
    batch_size: int = 128,
    num_workers: int = 8
):
    """
    JAX加速版本的norm统计计算
    
    Args:
        config_name: 配置名称
        max_frames: 最大处理帧数
        sample_ratio: 采样比例 (0.0-1.0)
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
    """
    print(f"使用JAX计算norm统计，可用设备: {jax.devices()}")
    
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    print(f"数据集总大小: {num_frames:,} 个样本")
    
    if sample_ratio < 1.0:
        num_frames = int(num_frames * sample_ratio)
        print(f"采样后大小: {num_frames:,} 个样本 (采样比例: {sample_ratio:.1%})")
    
    shuffle = False
    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True
        print(f"限制处理数量: {num_frames:,} 个样本")

    # 计算正确的批次数
    total_batches = (num_frames + batch_size - 1) // batch_size
    print(f"预期批次数: {total_batches}, 每批次大小: {batch_size}")
    
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=total_batches,  # 修复：使用批次数而不是样本数
    )
    
    def get_val(d, key):
        for k in key.split('.'):
            try:
                d = d[k] if isinstance(d, dict) else d[int(k)]
            except (KeyError, IndexError, ValueError, TypeError) as e:
                raise KeyError(f"获取 '{key}' 失败，在层级 '{k}' 出错：{e}")
        return np.asarray(d)
    
    def preprocess_batch(batch):
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
    
    # 创建JAX统计对象
    stats = {}
    for key in keys:
        if key in ["state", "actions"]:
            stats[key] = JAXRunningStats(exclude_dims=gripper_dims)
            print(f"为 '{key}' 创建JAX统计对象，跳过夹爪维度: {gripper_dims}")
        else:
            stats[key] = JAXRunningStats()
    
    print(f"开始JAX加速计算归一化统计信息...")
    print(f"配置: batch_size={batch_size}, num_workers={num_workers}")
    
    start_time = time.time()
    processed_samples = 0
    total_data_points = 0
    total_batches = (num_frames + batch_size - 1) // batch_size
    
    try:
        for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, total=total_batches, desc="Computing stats with JAX")):
            # 添加提前退出机制以防万一
            if processed_samples >= num_frames:
                print(f"已达到预期样本数 {num_frames}，提前退出")
                break
                
            batch_processed = preprocess_batch(batch)
            batch.update(batch_processed)
            
            current_batch_size = len(batch[keys[0]])
            processed_samples += current_batch_size
            
            # 添加安全检查
            if processed_samples > num_frames * 1.1:  # 超出10%就警告
                print(f"警告: 处理的样本数 ({processed_samples}) 远超预期 ({num_frames})")
            
            for key in keys:
                values = np.asarray(batch[key])
                
                if batch_idx == 0:
                    print(f"'{key}' 批次维度: {values.shape}, 数据类型: {values.dtype}")
                    if key in ["state", "actions"]:
                        print(f"  -> JAX将跳过夹爪维度: {gripper_dims}")
                
                if values.ndim == 3:
                    values_reshaped = values.reshape(-1, values.shape[-1])
                    if key == keys[0]:
                        total_data_points += values_reshaped.shape[0]
                elif values.ndim == 2:
                    values_reshaped = values
                    if key == keys[0]:
                        total_data_points += values_reshaped.shape[0]
                else:
                    print(f"警告: '{key}' 的维度不符合预期: {values.shape}")
                    continue
                
                # JAX统计更新
                stats[key].update(values_reshaped)
            
            if (batch_idx + 1) % 50 == 0:  # JAX更快，更频繁地报告
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
        if stat.running_count > 0:
            stat_result = stat.get_statistics()
            norm_stats[key] = stat_result
            print(f"'{key}' JAX统计信息:")
            print(f"  样本数: {stat.running_count:,}")
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
        print(f"JAX总计算时间: {elapsed/60:.1f} 分钟")
        print(f"实际处理样本数: {processed_samples:,}")
        print(f"统计计算数据点数: {total_data_points:,}")
        print(f"平均样本处理速度: {processed_samples/elapsed:.1f} samples/s")
        print(f"平均数据点处理速度: {total_data_points/elapsed:.1f} points/s")
    else:
        print("错误: 没有收集到任何统计信息！")

if __name__ == "__main__":
    tyro.cli(main)