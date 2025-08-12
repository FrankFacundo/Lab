import json
import os

import cv2
from docling import (
    LayoutPredictor,  # Adjust this import based on your docling installation
)
from labelbox import Client


def normalize_bbox(bbox, image_width, image_height):
    """
    Convert an absolute bbox [x, y, w, h] into normalized coordinates [x_norm, y_norm, w_norm, h_norm].
    """
    x, y, w, h = bbox
    return [x / image_width, y / image_height, w / image_width, h / image_height]


def predict_layout(image_path):
    """
    Loads the image from image_path and returns predicted layout bounding boxes
    using the docling model.

    Returns:
        predictions: A list of dictionaries.
                     Each dict should have at least:
                       - 'bbox': [x, y, w, h] in pixel units
                       - Optional: 'label' and 'score'
    """
    # Load image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    height, width, _ = image.shape

    # Initialize the docling layout predictor (modify parameters as needed)
    predictor = LayoutPredictor()

    # Get predictions (the API below is assumed; update if your usage differs)
    predictions = predictor.predict(image)

    # Normalize bounding boxes relative to image dimensions
    normalized_predictions = []
    for pred in predictions:
        # Expect each pred to have a key 'bbox' as [x, y, w, h]
        norm_bbox = normalize_bbox(pred["bbox"], width, height)
        normalized_predictions.append(
            {
                "name": pred.get("label", "layout"),  # default label "layout"
                "bbox": norm_bbox,
                "score": pred.get("score", 1.0),
            }
        )

    return normalized_predictions, os.path.basename(image_path)


def upload_predictions_to_labelbox(
    api_key, model_version_id, data_row_id, external_id, result
):
    """
    Uploads predictions to Labelbox using the createExternalPredictions GraphQL mutation.

    Args:
        api_key (str): Your Labelbox API key.
        model_version_id (str): The Labelbox model version ID associated with these predictions.
        data_row_id (str): The DataRow ID for the image.
        external_id (str): The external identifier for the image (typically its filename).
        result (dict): A dictionary containing the prediction result (e.g., {"objects": [...]})
    """
    client = Client(api_key=api_key)

    mutation = """
    mutation createExternalPredictions($externalPredictions: ExternalPredictionImport!) {
      createExternalPredictions(data: $externalPredictions) {
        id
      }
    }
    """

    external_predictions = {
        "modelVersionId": model_version_id,
        "predictions": [
            {
                "dataRowId": data_row_id,
                "externalId": external_id,
                # Labelbox expects the result to be a JSON-encoded string.
                "result": json.dumps(result),
            }
        ],
    }

    variables = {"externalPredictions": external_predictions}
    response = client.execute(mutation, variables)
    print("Prediction upload response:", response)


def main():
    # ====== CONFIGURATION ======
    # Path to the image you want to process.
    image_path = "path/to/your/image.jpg"

    # Labelbox credentials and IDs.
    LABELBOX_API_KEY = os.environ.get("LABELBOX_API_KEY", "your_labelbox_api_key")
    MODEL_VERSION_ID = (
        "your_model_version_id"  # Replace with your model version ID in Labelbox.
    )
    DATA_ROW_ID = (
        "your_data_row_id"  # Replace with the DataRow ID for this image in Labelbox.
    )
    # ============================

    # Run docling layout prediction.
    try:
        predictions, external_id = predict_layout(image_path)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Prepare result in the format expected by Labelbox.
    # For example, Labelbox might expect a JSON object with a key "objects" that is a list of predicted items.
    result = {"objects": predictions}

    # Upload predictions to Labelbox.
    try:
        upload_predictions_to_labelbox(
            api_key=LABELBOX_API_KEY,
            model_version_id=MODEL_VERSION_ID,
            data_row_id=DATA_ROW_ID,
            external_id=external_id,
            result=result,
        )
    except Exception as e:
        print(f"Error uploading predictions to Labelbox: {e}")


if __name__ == "__main__":
    main()
