from pprint import pprint
from pathlib import Path
from typing import Iterable, Optional, Union
import os
import numpy as np
import requests
import torch
from dataset_cppe5 import CPPE5Dataset
from datasets import load_dataset, load_from_disk
from image_transform import train_augmentation_and_transform, validation_transform
from mape_evaluator import MAPEvaluator
from PIL import Image, ImageDraw
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)
from transformers import RTDetrForObjectDetection


def main(mode):
    if mode == "train":
        train()

    if mode == "inference":
        inference()


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


def train():
    checkpoint = "PekingU/rtdetr_r50vd_coco_o365"
    image_size = 480

    dataset = get_dataset()

    dataset = dataset.filter(all_bboxes_valid)

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
    # id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    # show_sample(dataset, id2label)

    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"width": image_size, "height": image_size},
    )

    # show_transformed_image(dataset, id2label)
    train_dataset = CPPE5Dataset(
        dataset["train"], image_processor, transform=train_augmentation_and_transform
    )
    validation_dataset = CPPE5Dataset(
        dataset["validation"], image_processor, transform=validation_transform
    )
    test_dataset = CPPE5Dataset(
        dataset["test"], image_processor, transform=validation_transform
    )

    train_dataset[15]
    # show_transformed_image_2(train_dataset, id2label)

    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=image_processor, threshold=0.01, id2label=id2label
    )

    # model = AutoModelForObjectDetection.from_pretrained(
    #     checkpoint,
    #     id2label=id2label,
    #     label2id=label2id,
    #     anchor_image_size=None,
    #     ignore_mismatched_sizes=True,
    # )

    _model_path = "model_artifacts/layout"
    artifacts_path = download_models(".", progress=True) / _model_path
    model_config = os.path.join(str(artifacts_path), "config.json")
    model = RTDetrForObjectDetection.from_pretrained(
        str(artifacts_path), config=model_config
    ).to("cuda")

    training_args = TrainingArguments(
        output_dir="rtdetr-exams-finetune",
        num_train_epochs=10,
        max_grad_norm=0.1,
        learning_rate=5e-5,
        warmup_steps=300,
        per_device_train_batch_size=32,
        dataloader_num_workers=4,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()

    # Evaluate
    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="eval")
    pprint(metrics)


def inference():
    # Inference

    device = "cuda"

    local_image_path = "test.png"
    image = Image.open(local_image_path).convert("RGB")

    model_repo = "./rtdetr-exams-finetune/checkpoint-260"

    image_processor = AutoImageProcessor.from_pretrained(model_repo)

    _model_path = "./rtdetr-exams-finetune/checkpoint-260"
    model_config = os.path.join(str(_model_path), "config.json")
    model = RTDetrForObjectDetection.from_pretrained(
        str(_model_path), config=model_config
    ).to("cuda")

    ## Detect bounding boxes
    inputs = image_processor(images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])

    result = image_processor.post_process_object_detection(
        outputs, threshold=0.4, target_sizes=target_sizes
    )[0]

    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    ## Plot the result
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="red")

    image_with_boxes.show()


def get_dataset():
    # dataset = load_dataset("cppe-5")
    dataset = load_from_disk("dataset/dataset_exams")

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(0.15, seed=1337)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    return dataset


def all_bboxes_valid(example):
    img_width, img_height = example["image"].size  # Get image dimensions

    for bbox in example["objects"]["bbox"]:  # Iterate over all bounding boxes
        (
            x_min,
            y_min,
            width,
            height,
        ) = bbox  # Assuming bbox format is [x_min, y_min, width, height]
        x_max = x_min + width
        y_max = y_min + height

        # If any bbox is out of bounds, return False
        if not (
            0 <= x_min < img_width
            and 0 <= x_max <= img_width
            and 0 <= y_min < img_height
            and 0 <= y_max <= img_height
        ):
            return False  # Discard this image

    return True  # Keep this image


def show_sample(dataset, id2label):
    # Load image and annotations
    image = dataset["train"][65]["image"]
    annotations = dataset["train"][65]["objects"]

    # Draw bounding boxes and labels
    draw = ImageDraw.Draw(image)
    for i in range(len(annotations["id"])):
        box = annotations["bbox"][i]
        class_idx = annotations["category"][i]
        x, y, w, h = tuple(box)
        draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
        draw.text((x, y), id2label[class_idx], fill="white")

    image.show()


def show_transformed_image(dataset, id2label):
    for i in [65]:
        image = dataset["train"][i]["image"]
        annotations = dataset["train"][i]["objects"]

        # Apply the augmentation
        output = train_augmentation_and_transform(
            image=np.array(image),
            bboxes=annotations["bbox"],
            category=annotations["category"],
        )

        # Unpack the output
        image = Image.fromarray(output["image"])
        categories, boxes = output["category"], output["bboxes"]

        # Draw the augmented image
        draw = ImageDraw.Draw(image)
        for category, box in zip(categories, boxes):
            print(box)
            x, y, w, h = box
            draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
            draw.text((x, y), id2label[category], fill="white")
        image.show()


def show_transformed_image_2(train_dataset, id2label):
    for i in [65]:
        sample = train_dataset[i]

        # De-normalize image
        image = sample["pixel_values"]
        print("Image tensor shape:", image.shape)
        image = image.numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        image = Image.fromarray(image.astype(np.uint8))

        # Convert boxes from [center_x, center_y, width, height] to [x, y, width, height] for visualization
        boxes = sample["labels"]["boxes"].numpy()
        print("Boxes shape:", boxes.shape)
        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        w, h = image.size
        boxes = boxes * np.array([w, h, w, h])[None]

        categories = sample["labels"]["class_labels"].numpy()
        print("Categories shape:", categories.shape)

        # Draw boxes and labels on image
        draw = ImageDraw.Draw(image)
        for box, category in zip(boxes, categories):
            print(box)
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
            draw.text((x, y), id2label[category], fill="white")
        image.show()


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


if __name__ == "__main__":
    main(mode="train")
