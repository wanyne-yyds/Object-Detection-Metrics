import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from lib.utils import xyxyxyxy2xywhr, xywhr2xyxyxyxy

def generate_obb_labels(image, points, show=False):
    polylines_bbox = []
    polylines_new_bbox = []
    labels_list = []
    for obj in points:
        labels = obj["label"]
        labels = labels.split("_front")[0]
        bbox = np.array(obj["points"], dtype=np.int32).flatten()[np.newaxis, :]
        xywhr = xyxyxyxy2xywhr(bbox)
        new_bbox = xywhr2xyxyxyxy(xywhr).reshape(len(bbox), -1, 2).astype(np.int32)
        bbox = bbox.reshape(len(bbox), -1, 2).astype(np.int32)
        polylines_bbox.append(bbox)
        polylines_new_bbox.append(new_bbox)
        labels_list.append([labels, new_bbox.flatten().tolist()])

        if show:
            cv2.polylines(image, bbox, isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.polylines(image, new_bbox, isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(image, "Src", (bbox[0][0][0], bbox[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Gen", (new_bbox[0][0][0], new_bbox[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image, labels_list

if __name__ == '__main__':
    files_dir = r"Z:\dataset\Motorcycle_Stop_Line_Dataset\Src\Labels\Test"
    output = Path(files_dir + "_obb_labels")
    output.mkdir(exist_ok=True, parents=True)

    show_img = True

    for img_file in Path(files_dir).rglob("*.*g"):
        json_file = img_file.with_suffix(".json")
        output_son = output / json_file.parts[-2]
        output_son.mkdir(exist_ok=True, parents=True)

        if not json_file.exists():
            with open(output_son / f"{json_file.stem}.txt", "w") as file:
                file.write("")
            continue
        
        image = cv2.imread(str(img_file))
        with open(json_file, "r") as file:
            data = json.load(file)
            image, labels_list = generate_obb_labels(image, data["shapes"], show=show_img)

            if show_img:
                cv2.imshow("image", image)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
            else:
                with open(output_son / f"{json_file.stem}.txt", "w") as file:
                    for label in labels_list:
                        class_name, points = label
                        points = " ".join(map(str, points))
                        file.write(f"{class_name} {points}")