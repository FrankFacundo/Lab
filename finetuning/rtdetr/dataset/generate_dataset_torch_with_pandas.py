import polars as pl
import pandas as pd
import io
import base64
from PIL import Image
from datasets import Dataset, DatasetDict

# Load the Parquet file
df = pl.read_parquet("output_annotations.parquet").to_pandas()

# Generate a mapping from category names to integer IDs
id2label = {
    0: "background",
    1: "caption",
    2: "footnote",
    3: "formula",
    4: "list_item",
    5: "page_footer",
    6: "page_header",
    7: "picture",
    8: "section_header",
    9: "table",
    10: "text",
    11: "title",
    12: "document_index",
    13: "code",
    14: "checkbox_selected",
    15: "checkbox_unselected",
    16: "form",
    17: "key_value_region",
}
label_to_id = {v: k for k, v in id2label.items()}


print(label_to_id)


# Function to convert base64 to PIL Image
def decode_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return image


# Generate dataset in the required format
dataset_list = []
for idx, row in df.iterrows():
    image = decode_image(row["image_base64"])
    width, height = image.size

    objects = {
        "id": list(range(len(row["bboxes"]))),  # Assigning unique IDs to objects
        "area": [
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in row["bboxes"]
        ],  # Calculate area
        "bbox": [
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            for bbox in row["bboxes"]
        ],  # Convert to xywh
        "category": [
            label_to_id[label] - 1 for label in row["labels"]
        ],  # Integer category
        "category_name": row["labels"],  # Category name
    }

    dataset_list.append(
        {
            "image_id": idx,
            "image": image,
            "width": width,
            "height": height,
            "objects": objects,
        }
    )

# Convert to Hugging Face dataset
dataset_pandas = pd.DataFrame(dataset_list)
hf_dataset = Dataset.from_dict(
    dataset_pandas.reset_index(drop=True).to_dict(orient="list")
)

# If needed, split into train/test
split_ratio = 0.9
train_size = int(len(hf_dataset) * split_ratio)
hf_dataset = hf_dataset.train_test_split(train_size=train_size)

# Convert to DatasetDict format
hf_dataset = DatasetDict({"train": hf_dataset["train"], "test": hf_dataset["test"]})

# Save the dataset
hf_dataset.save_to_disk("dataset_exams_2")

print(hf_dataset)
