import os
import sys

import labelbox as lb


def get_image_file_paths(directory):
    """
    Recursively traverse the directory and return a list of image file paths.
    Supports common image extensions.
    """
    supported_extensions = ".jpg"
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths[:]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_images.py /path/to/image/directory")
        sys.exit(1)

    # Directory provided as a parameter
    directory = sys.argv[1]
    local_file_paths = get_image_file_paths(directory)

    # Optional: Limit to 15k files if more images are found
    max_files = 15000
    if len(local_file_paths) > max_files:
        print(
            f"Found {len(local_file_paths)} images. Limiting upload to the first {max_files} files."
        )
        local_file_paths = local_file_paths[:max_files]

    # Create a new dataset on Labelbox
    client = lb.Client(os.getenv("LABELBOX_TOKEN"))

    dataset = client.get_dataset("cm7cb7qka005v0773o543dsko")

    try:
        # Create data rows from the local image paths
        task = dataset.create_data_rows(local_file_paths)
        task.wait_till_done()
        print("Upload completed successfully.")
    except Exception as err:
        print(f"Error while creating labelbox dataset - Error: {err}")
