import os
import cv2
import json
import numpy as np
from pathlib import Path

def draw_half_rectangles(image, points):
    for obj in points:
        labels = obj["label"]
        bbox = np.array(obj["points"], np.int32).flatten()
        x0, y0, x1, y1, x2, y2, x3, y3 = bbox
        cv2.polylines(image, [np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(image, labels, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    json_folder = r"Z:\dataset\Motorcycle_Stop_Line_Dataset\Src\Labels\Test"
    image_output = json_folder + "_forgery"
    os.makedirs(image_output, exist_ok=True)

    for json_file in Path(json_folder).rglob("*.json"):
        image_path = json_file.with_suffix(".jpg")
        if not image_path.exists():
            image_path = json_file.with_suffix(".png")

        image = cv2.imread(str(image_path))
        with open(json_file, "r") as file:
            data = json.load(file)
            image = draw_half_rectangles(image, data["shapes"])

            # cv2.imwrite(str(Path(image_output) / image_path.name), image)

            cv2.imshow("image", image)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break