import io
import os
import sys

import polars as pl
from PIL import Image


def export_images(parquet_file):
    # Define the output directory
    output_dir = "images_from_parquet"
    os.makedirs(output_dir, exist_ok=True)

    # Read the Parquet file into a Polars DataFrame
    df = pl.read_parquet(parquet_file)

    # (Optional) Print the first few rows to inspect the DataFrame
    print(df.head())

    # Convert the DataFrame into a list of dictionaries for easier iteration
    data_rows = df.to_dicts()

    # Iterate over each row, extract the image, and save it
    for i, row in enumerate(data_rows):
        # Retrieve the image bytes from the "image" column
        image_bytes = row["image"]

        # Open the image using Pillow
        image = Image.open(io.BytesIO(image_bytes))

        # Define the file name; if you have a "global_key" in your data, you can use it instead
        file_name = f"image_{i:05d}.jpg"
        file_path = os.path.join(output_dir, file_name)

        # Save the image as JPEG
        image.save(file_path)
        print(f"Saved {file_path}")

    print("All images have been exported.")


if __name__ == "__main__":
    parquet_file = sys.argv[1]
    export_images(parquet_file)
