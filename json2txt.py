import os
import json
from pathlib import Path

json_dir = r"E:\Code\rknn\rknn_model_zoo-v2.3.2\install\rv1126b_linux_armhf\BSD_Dataset"
output_txt = r"E:\Code\rknn\rknn_model_zoo-v2.3.2\install\rv1126b_linux_armhf\BSD_Dataset_txt"
os.makedirs(output_txt, exist_ok=True)

for json_file in Path(json_dir).rglob("*.json"):
    txt_name = json_file.with_suffix('.txt')
    txt_name = Path(output_txt) / txt_name.name
    # 读取 JSON 文件
    with open(str(json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取 label 和 points
    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label != "person":
            continue
        points = shape.get("points", [])
        if label and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            line = f"{label} {int(x1)} {int(y1)} {int(x2)} {int(y2)}"
            lines.append(line)

    # 写入 txt 文件
    with open(txt_name, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
