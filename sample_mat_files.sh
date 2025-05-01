#!/bin/bash

# --- 配置 ---
SOURCE_DIR="datasets/measured_hrrp"
DEST_DIR="datasets/measured_hrrp_"
NUM_FILES_TO_SAMPLE=3000

# --- 检查工具 ---
if ! command -v shuf &> /dev/null; then
    echo "错误: 'shuf' 命令未找到。请安装 coreutils 包 (通常默认安装在 Linux 系统中)。" >&2
    exit 1
fi

# --- 检查源目录 ---
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录 '$SOURCE_DIR' 不存在。" >&2
    exit 1
fi

# --- 创建目标目录 (如果不存在) ---
mkdir -p "$DEST_DIR"
if [ $? -ne 0 ]; then
    echo "错误: 无法创建目标目录 '$DEST_DIR'。" >&2
    exit 1
fi
echo "目标目录 '$DEST_DIR' 已准备好。"

# --- 查找并计数 .mat 文件 ---
# -maxdepth 1 假设文件直接在 SOURCE_DIR 中，而不是子目录
# -type f 只查找文件
echo "正在查找 '$SOURCE_DIR' 中的 .mat 文件..."
mapfile -t all_mat_files < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.mat")
actual_file_count=${#all_mat_files[@]}
echo "找到 $actual_file_count 个 .mat 文件。"

# --- 检查文件数量是否足够 ---
if [ "$actual_file_count" -lt "$NUM_FILES_TO_SAMPLE" ]; then
    echo "错误: 源目录中只有 $actual_file_count 个 .mat 文件，不足所需的 $NUM_FILES_TO_SAMPLE 个。" >&2
    exit 1
fi

# --- 随机抽取并复制 ---
echo "正在随机抽取 $NUM_FILES_TO_SAMPLE 个文件并复制到 '$DEST_DIR'..."

# 使用 printf 将数组元素按行打印，然后管道给 shuf
printf "%s\n" "${all_mat_files[@]}" | shuf -n "$NUM_FILES_TO_SAMPLE" | while IFS= read -r file_to_copy; do
    # 检查文件是否存在（以防万一）
    if [ -f "$file_to_copy" ]; then
        cp "$file_to_copy" "$DEST_DIR/"
        # 可以取消下面的注释来查看复制进度，但会输出很多行
        # echo "已复制: $(basename "$file_to_copy")"
    else
        echo "警告: 文件 '$file_to_copy' 在抽取后未找到，跳过。" >&2
    fi
done

# --- 检查复制是否成功 (简单检查文件数量) ---
copied_count=$(find "$DEST_DIR" -maxdepth 1 -type f -name "*.mat" | wc -l)
echo "检查目标目录 '$DEST_DIR'，发现 $copied_count 个 .mat 文件。"

if [ "$copied_count" -eq "$NUM_FILES_TO_SAMPLE" ]; then
    echo "成功复制 $NUM_FILES_TO_SAMPLE 个随机 .mat 文件到 '$DEST_DIR'。"
else
    # 如果数量不匹配，可能是在复制过程中出错或被中断
    echo "警告: 目标目录中的文件数量 ($copied_count) 与请求的数量 ($NUM_FILES_TO_SAMPLE) 不匹配。请检查复制过程。" >&2
fi

exit 0