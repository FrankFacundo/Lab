import os
import re
import uuid

import labelbox as lb
import labelbox.data.annotation_types as lb_types
import polars as pl
from labelbox import LabelImport

# Initialize Labelbox client
client = lb.Client(os.getenv("LABELBOX_TOKEN"))

# === Build a mapping from image number to data row id ===
# (This assumes your Labelbox project export has external_id filenames like "image_09935.jpg")
project = client.get_project("")
export_params = {"data_row_details": True}
filters = {}

task = project.export(params=export_params, filters=filters)
task.wait_till_done()

pattern = re.compile(r"image_(\d+)\.jpg")
export_dict = {}
for data_row in task.get_buffered_stream():
    external_id = data_row.json["data_row"]["external_id"]
    data_row_id = data_row.json["data_row"]["id"]
    filename = os.path.basename(external_id)
    match = pattern.search(filename)
    if match:
        image_number = int(match.group(1))  # e.g., "09935" becomes 9935
        export_dict[image_number] = data_row_id
    else:
        print(f"Warning: Could not extract image number from {filename}")

# === Read the Parquet file with prediction data ===
df = pl.read_parquet(
    "/home/frank/Code/multumbabel/lab/finetuning/rtdetr/dataset/doclaynet/results.parquet"
)
df = df[:5]  # Limit for testing purposes

# (Optional) Verify the contents
print(df.head())

predictions = []
# data_rows = df.to_dicts()

list_bboxes = df["bboxes"].to_list()
list_labels = df["label"].to_list()

# Iterate over each prediction row; assume the row index equals the image number.
for idx, (bboxes, labels) in enumerate(zip(list_bboxes, list_labels)):
    image_number = idx  # Using idx as the image number
    data_row_id = export_dict.get(image_number)
    if data_row_id is None:
        print(f"Warning: No data row id found for image number {image_number}")
        continue

    annotations = []
    for box, label in zip(bboxes, labels):
        annotation = lb_types.ObjectAnnotation(
            name=label,
            value=lb_types.Rectangle(
                start=lb_types.Point(x=box[0], y=box[1]),
                end=lb_types.Point(x=box[2], y=box[3]),
            ),
        )
        annotations.append(annotation)

    # Replace global_key with the corresponding data row id
    label = lb_types.Label(
        data={"global_key": data_row_id},
        annotations=annotations,
    )
    predictions.append(label)

print(f"Prepared {len(predictions)} predictions for upload.")


upload_job = LabelImport.create_from_objects(
    client=client,
    project_id="",
    name="label_import_job" + str(uuid.uuid4()),
    labels=predictions,
)


upload_job.wait_till_done(sleep_time_seconds=1, show_progress=True)
print("Errors:", upload_job.errors)
