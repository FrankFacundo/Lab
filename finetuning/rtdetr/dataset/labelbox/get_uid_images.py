import os
import re

import labelbox as lb

client = lb.Client(os.getenv("LABELBOX_TOKEN"))

project = client.get_project("")

export_params = {
    "data_row_details": True,
}
filters = {}

task = project.export(params=export_params, filters=filters)
task.wait_till_done()

# Define the regex pattern for extracting the image number.
pattern = re.compile(r"image_(\d+)\.jpg")

# Build a dictionary mapping image_number to data_row id.
export_dict = {}
for data_row in task.get_buffered_stream():
    external_id = data_row.json["data_row"]["external_id"]
    data_row_id = data_row.json["data_row"]["id"]
    filename = os.path.basename(external_id)

    match = pattern.search(filename)
    if match:
        image_number = int(match.group(1))  # converts "09935" to 9935
        export_dict[image_number] = data_row_id
    else:
        print(f"Warning: Could not extract image number from {filename}")

print(export_dict)
