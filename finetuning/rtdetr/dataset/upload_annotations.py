#!/usr/bin/env python
import os
import re
import uuid

import polars as pl
import requests
from tqdm import tqdm  # Added import for the progress bar

# Set Label Studio connection parameters (adjust as needed)
LABELSTUDIO_URL = os.getenv("LABELSTUDIO_URL", "http://localhost:8080")
API_KEY = os.getenv("LABELSTUDIO_API_KEY", "")
PROJECT_ID = os.getenv("LABELSTUDIO_PROJECT_ID", 3)

HEADERS = {"Authorization": f"Token {API_KEY}"}


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
                tasks_mapping[image_number] = task["id"]
            else:
                print(f"Warning: Could not extract image number from {filename}")
        page += 1
    return tasks_mapping


def upload_prediction(task_id, results):
    """
    Retrieve the task JSON and post the prediction result to Label Studio.
    """
    # Fetch the task's full JSON (needed by the prediction payload)
    # response = requests.get(f"{LABELSTUDIO_URL}/api/tasks/{task_id}", headers=HEADERS)
    # if not response.ok:
    #     print(f"Failed to get task {task_id}: {response.text}")
    #     return
    # task_json = response.json()

    payload = {
        "result": results,
        # "task": task_json["id"],
        "task": task_id,
        "score": 0,
        "model_version": "auto",
    }
    post_response = requests.post(
        f"{LABELSTUDIO_URL}/api/predictions/", headers=HEADERS, json=payload
    )
    if post_response.ok:
        print(f"Uploaded prediction for task {task_id}")
    else:
        print(f"Failed to upload prediction for task {task_id}: {post_response.text}")


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python upload_annotations.py /path/to/predictions.parquet")
    #     sys.exit(1)

    # parquet_path = sys.argv[1]
    parquet_path = "/home/frank/Code/multumbabel/lab/finetuning/rtdetr/dataset/results_unmsm_2025.parquet"
    df = pl.read_parquet(parquet_path)
    # df = df.head(5)  # Limit for testing purposes; remove or adjust as needed

    # Build a mapping from image number (extracted from filename) to task id.
    tasks_mapping = get_tasks_mapping()
    print(f"Fetched mapping for {len(tasks_mapping)} tasks.")

    # Assumes the DataFrame has "bboxes" and "label" columns
    list_bboxes = df["bboxes"].to_list()
    list_labels = df["label"].to_list()

    # Iterate over each prediction row (assuming the row index equals the image number)
    for idx, (bboxes, labels) in tqdm(
        enumerate(zip(list_bboxes, list_labels)),
        total=len(list_bboxes),
        desc="Processing predictions",
    ):
        task_id = tasks_mapping.get(idx)
        if task_id is None:
            print(f"Warning: No task id found for image number {idx}")
            continue

        # Retrieve the image path from the task to compute image dimensions
        task_response = requests.get(
            f"{LABELSTUDIO_URL}/api/tasks/{task_id}", headers=HEADERS
        )
        if not task_response.ok:
            print(
                f"Failed to get task {task_id} for image dimensions: {task_response.text}"
            )
            continue
        task = task_response.json()
        image_path = task.get("data", {}).get("image")
        img_width, img_height = 468, 615
        # try:
        #     with Image.open(image_path) as img:
        #         img_width, img_height = img.size
        # except Exception as e:
        #     print(f"Could not open image {image_path}: {e}")
        #     continue

        results = []
        # Process each bounding box and its corresponding label.
        # Assumes each box is [x_min, y_min, x_max, y_max] in absolute pixels.
        for box, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = box
            # Convert to percentages (Label Studio expects percentages)
            x = (x_min / img_width) * 100
            y = (y_min / img_height) * 100
            width_percent = ((x_max - x_min) / img_width) * 100
            height_percent = ((y_max - y_min) / img_height) * 100

            result_item = {
                "id": str(uuid.uuid4()),
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": x,
                    "y": y,
                    "width": width_percent,
                    "height": height_percent,
                    "rotation": 0,
                    "rectanglelabels": [label],
                },
            }
            results.append(result_item)

        if results:
            upload_prediction(task_id, results)
