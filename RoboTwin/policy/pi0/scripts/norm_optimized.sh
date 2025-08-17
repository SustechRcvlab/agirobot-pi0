#!/bin/bash

echo "=== 优化版本norm计算脚本 ==="
echo "1. 快速测试模式 (采样10%数据)"
echo "2. 中等测试模式 (采样50%数据)" 
echo "3. 完整计算模式 (100%数据)"
echo "4. 取消当前运行"

read -p "请选择模式 (1-4): " choice

case $choice in
    1)
        echo "启动快速测试模式..."
        uv run scripts/compute_norm_stats_optimized.py \
            --config-name pi0_base_agibot_lora \
            --sample-ratio 0.1 \
            --batch-size 128 \
            --num-workers 24
        ;;
    2)
        echo "启动中等测试模式..."
        uv run scripts/compute_norm_stats_optimized.py \
            --config-name pi0_base_agibot_lora \
            --sample-ratio 0.5 \
            --batch-size 128 \
            --num-workers 24
        ;;
    3)
        echo "启动完整计算模式..."
        uv run scripts/compute_norm_stats_optimized.py \
            --config-name pi0_base_agibot_lora \
            --sample-ratio 1.0 \
            --batch-size 128 \
            --num-workers 24
        ;;
    4)
        echo "正在查找并终止当前的norm计算进程..."
        pkill -f "compute_norm_stats"
        echo "已终止相关进程"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
