import os
import uuid

# Labelbox imports
import labelbox as lb
import labelbox.data.annotation_types as lb_types
import polars as pl
from labelbox import MALPredictionImport

# Initialize Labelbox client
client = lb.Client(os.getenv("LABELBOX_TOKEN"))

# Read the Parquet file into a Polars DataFrame
df = pl.read_parquet(
    "/home/frank/Code/multumbabel/lab/finetuning/rtdetr/dataset/doclaynet/results.parquet"
)
df = df[:5]

# (Optional) Print the first few rows to verify the contents
print(df.head())

# Prepare a list to hold Labelbox prediction objects
predictions = []

# Convert the DataFrame to a list of dictionaries for easier iteration
data_rows = df.to_dicts()

for idx, row in enumerate(data_rows):
    # Get or generate a global key for the Labelbox data row.
    # If your Parquet already includes a "global_key" column, it will be used.
    global_key = row.get("global_key") or str(uuid.uuid4())

    # Retrieve bounding boxes from the row (assumed format: list of [x1, y1, x2, y2])
    bboxes = row["bboxes"]

    # Create an annotation for each bounding box.
    # Adjust the "name" parameter if you have different object types.
    annotations = []
    for box in bboxes:
        annotation = lb_types.ObjectAnnotation(
            name="bbox",  # or use a more descriptive name if needed
            value=lb_types.Rectangle(
                start=lb_types.Point(x=box[0], y=box[1]),
                end=lb_types.Point(x=box[2], y=box[3]),
            ),
        )
        annotations.append(annotation)

    # Create a Labelbox label object using the global key and its annotations.
    label = lb_types.Label(
        data={"global_key": global_key},
        annotations=annotations,
    )
    predictions.append(label)

print(f"Prepared {len(predictions)} predictions for upload.")

# Upload the predictions to Labelbox using MALPredictionImport.
# Replace the project_id with your Labelbox project id.
upload_job = MALPredictionImport.create_from_objects(
    client=client,
    project_id="cm72frjaw02lm070e975w9od3",  # <-- change to your project id
    name="mal_job_" + str(uuid.uuid4()),
    predictions=predictions,
)

# Wait until the job completes
upload_job.wait_till_done()

# Print any errors that occurred during the upload
print("Errors:", upload_job.errors)
