#!/usr/bin/env python
import io
import os
import re

import polars as pl
import requests
from PIL import Image
from tqdm import tqdm  # For showing progress

# check https://labelstud.io/guide/export


# Set Label Studio connection parameters (adjust as needed)
LABELSTUDIO_URL = os.getenv("LABELSTUDIO_URL", "http://localhost:8080")
API_KEY = os.getenv("LABELSTUDIO_API_KEY", "")
PROJECT_ID = os.getenv("LABELSTUDIO_PROJECT_ID", 3)

HEADERS = {"Authorization": f"Token {API_KEY}"}
IMAGE_WIDTH = 468
IMAGE_HEIGHT = 615


def get_tasks_mapping():
    """
    Fetch all tasks in the project and create a mapping from image number
    (extracted from the filename in the task's data) to the Label Studio task id.
    Assumes filenames like "image_09935.jpg".
    """
    tasks_mapping = {}
    page = 1
    page_size = 100
    while True:
        response = requests.get(
            f"{LABELSTUDIO_URL}/api/projects/{PROJECT_ID}/tasks/",
            headers=HEADERS,
            params={"page": page, "page_size": page_size},
        )
        if not response.ok:
            print("Failed to fetch tasks:", response.text)
            break
        tasks = response.json()
        if not tasks:
            break
        for task in tasks:
            image_url = task.get("data", {}).get("image", "")
            filename = os.path.basename(image_url)
            match = re.search(r"image_(\d+)\.jpg", filename)
            if match:
                image_number = int(match.group(1))
                tasks_mapping[image_number] = {
                    "task_id": task["id"],
                    "image_url": image_url,
                }
            else:
                print(f"Warning: Could not extract image number from {filename}")
        page += 1
    return tasks_mapping


def encode_image_base64(image_url):
    """
    Reads the image from the specified path and returns it as binary data.
    """
    try:
        # Extract the actual file path from the image_url
        parsed_path = image_url.replace("/data/local-files/?d=", "")

        with Image.open(parsed_path) as image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            return img_bytes
    except Exception as e:
        print(f"Error reading image {parsed_path}: {e}")
        return None


def save_bboxes_to_parquet(tasks_mapping, output_file):
    """
    Save bounding boxes and labels to a Parquet file.
    """
    bbox_data = []

    # Fetch task details and bounding boxes
    for image_number, task_info in tqdm(
        tasks_mapping.items(), desc="Downloading annotations"
    ):
        task_id = task_info["task_id"]
        image_url = task_info["image_url"]

        task_response = requests.get(
            f"{LABELSTUDIO_URL}/api/tasks/{task_id}", headers=HEADERS
        )
        if not task_response.ok:
            print(f"Failed to get task {task_id}: {task_response.text}")
            continue

        task_data = task_response.json()
        image_data = task_data.get("data", {}).get("image", "")
        annotations = task_data.get("annotations", [])
        if annotations:
            bboxes_label_studio = annotations[0].get("result", [])
        else:
            continue

        bboxes = []
        labels = []

        for bbox in bboxes_label_studio:
            if bbox["type"] == "rectanglelabels":
                label = bbox["value"]["rectanglelabels"][0]
                # x_min = bbox["value"]["x"]image_width
                # y_min = bbox["value"]["y"]
                # width = bbox["value"]["width"]
                # height = bbox["value"]["height"]
                x_min = (bbox["value"]["x"] / 100) * IMAGE_WIDTH
                y_min = (bbox["value"]["y"] / 100) * IMAGE_HEIGHT
                width = (bbox["value"]["width"] / 100) * IMAGE_WIDTH
                height = (bbox["value"]["height"] / 100) * IMAGE_HEIGHT

                x_max = x_min + width
                y_max = y_min + height

                # bboxes.append([x_min, y_min, width, height])
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(label)

        image_base64 = encode_image_base64(image_url)

        bbox_data.append(
            {
                "image_path": image_url,
                "image_base64": image_base64,
                "bboxes": bboxes,
                "labels": labels,
            }
        )

    # Create a Polars DataFrame and save to Parquet
    df = pl.DataFrame(bbox_data)
    df.write_parquet(output_file)
    print(f"Saved annotations to {output_file}")


if __name__ == "__main__":
    output_file = "output_annotations.parquet"

    # Build a mapping from image number to task id
    tasks_mapping = get_tasks_mapping()
    print(f"Fetched mapping for {len(tasks_mapping)} tasks.")

    # Save the bounding boxes and labels to Parquet
    save_bboxes_to_parquet(tasks_mapping, output_file)
