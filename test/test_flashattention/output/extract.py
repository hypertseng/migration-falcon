import re
import os
from collections import defaultdict


def extract_module_time_total(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 使用正则表达式匹配 module_total_time
    pattern = re.compile(r"Module:Kernel, total_time:(\d+)us")
    matches = pattern.findall(content)

    # 输出所有匹配到的 module_total_time
    for match in matches:
        print(match, end=",")


def get_file_number(file_name):
    # 提取文件名中的数字部分
    match = re.search(r"(\d+)(?=\D*$)", file_name)
    return int(match.group(1)) if match else float("inf")


def get_file_prefix(file_name):
    # 提取文件名前缀（去掉最后的数字部分）
    match = re.search(r"^(.*?)(\d+)?\D*$", file_name)
    return match.group(1) if match else file_name


# 遍历目录并按文件名前缀分组
file_groups = defaultdict(list)
for root, dirs, files in os.walk(
    r"C:/Users/13706/Desktop/migration-falcon/test/test_flashattention/output/forward"
):
    for file in files:
        prefix = get_file_prefix(file)
        file_groups[prefix].append(os.path.join(root, file))

# 对每个组内的文件按结尾数字大小排序并处理
for prefix, file_list in file_groups.items():
    sorted_files = sorted(file_list, key=lambda x: get_file_number(os.path.basename(x)))
    for file_path in sorted_files:
        extract_module_time_total(file_path)
