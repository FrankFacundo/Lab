from pathlib import Path
from typing import Optional

from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from PIL import Image


class LayoutModel(object):
    _model_repo_folder = "ds4sd--docling-models"
    _model_path = "model_artifacts/layout"

    def __init__(self):
        artifacts_path = self.download_models() / self._model_path

        self.layout_predictor = LayoutPredictor(
            artifact_path=str(artifacts_path),
            device="cpu",
            num_threads=4,
        )

    def inference(self):
        # Inference

        # device = "cpu"

        local_image_path = "sample.png"
        image = Image.open(local_image_path)

        pred_items = {}
        for ix, pred_item in enumerate(self.layout_predictor.predict(image)):
            # pred_item example: {'l': 889.105712890625, 't': 832.9788208007812, 'r': 1623.8914794921875, 'b': 1111.317626953125, 'label': 'Text', 'confidence': 0.9830111265182495}
            pred_items[ix] = pred_item

        self.draw_image_and_pred_items(
            image,
            pred_items,
        )

    def draw_image_and_pred_items(self, image, pred_items):
        """
        Draws a page image that includes label names and confidence scores for each label.
        For each prediction, it draws a bounding box and annotates it with the label and confidence.
        """
        from PIL import ImageDraw, ImageFont

        # Create a copy of the image to preserve the original.
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)

        # Optionally load a TrueType font; fall back to the default if not available.
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        for item in pred_items.values():
            # Extract the bounding box coordinates, label, and confidence.
            l = item.get("l", 0)
            t = item.get("t", 0)
            r = item.get("r", 0)
            b = item.get("b", 0)
            label = item.get("label", "N/A")
            confidence = item.get("confidence", 0.0)

            # Round the coordinates for neatness.
            box = [round(l, 2), round(t, 2), round(r, 2), round(b, 2)]
            x, y, x2, y2 = tuple(box)

            # Draw the bounding box.
            draw.rectangle((x, y, x2, y2), outline="red", width=1)

            # Create the label text with confidence.
            text = f"{label} ({confidence:.2f})"
            draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)

        # Show the final annotated image.
        image_with_boxes.show()

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        if not progress:
            disable_progress_bars()
        download_path = snapshot_download(
            repo_id="ds4sd/docling-models",
            force_download=force,
            local_dir=local_dir,
            revision="v2.1.0",
        )

        return Path(download_path)


def main():
    layout_model = LayoutModel()
    layout_model.inference()


if __name__ == "__main__":
    main()
