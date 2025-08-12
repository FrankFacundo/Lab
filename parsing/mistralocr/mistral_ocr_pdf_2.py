import json
import base64
import shutil
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
import os
import re

api_key = os.getenv("MISTRAL_OCR")

client = Mistral(api_key=api_key)

# Path configuration
INPUT_DIR = Path(
    "pdfs_to_process"
)  # Folder where the user places the PDFs to be processed
DONE_DIR = Path("pdfs_done")  # Folder where processed PDFs will be moved
OUTPUT_ROOT_DIR = Path("ocr_output")  # Root folder for conversion results

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
DONE_DIR.mkdir(exist_ok=True)
OUTPUT_ROOT_DIR.mkdir(exist_ok=True)


def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    This converts base64 encoded images directly in the markdown...
    And replaces them with links to external images, so the markdown is more readable and organized.
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str


def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Part of the response from the Mistral API, which is an OCRResponse object...
    And returns a single string with the combined markdown of all the pages of the PDF.
    """
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)


def process_pdf(pdf_path: Path):
    # Process all PDFs in INPUT_DIR
    # - Important to be careful with the number of PDFs, as the Mistral API has a usage limit
    #   and it could cause errors by exceeding the limit.

    # PDF base name
    pdf_base = pdf_path.stem
    print(f"Processing {pdf_path.name} ...")

    # Output folders
    output_dir = OUTPUT_ROOT_DIR / pdf_base
    output_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # PDF -> OCR
    # with open(pdf_path, "rb") as f:
    #     pdf_bytes = f.read()

    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_path.name,
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr",
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    ocr_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=True,
    )

    # Save OCR in JSON
    # (in case something fails it could be reused, but it is not used in the rest of the code)
    ocr_json_path = output_dir / "ocr_response.json"
    with open(ocr_json_path, "w", encoding="utf-8") as json_file:
        json.dump(ocr_response.dict(), json_file, indent=4, ensure_ascii=False)
    print(f"OCR response saved in {ocr_json_path}")

    # OCR -> Markdown prepared for Obsidian
    # - That is, from base64 encoded images, it converts them to links to
    #   external images and generates the images as such, in a subfolder.

    global_counter = 1
    updated_markdown_pages = []

    for page in ocr_response.pages:
        updated_markdown = page.markdown
        for image_obj in page.images:
            # base64 to image
            base64_str = image_obj.image_base64
            if base64_str.startswith("data:"):
                base64_str = base64_str.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_str)

            # image extensions
            ext = Path(image_obj.id).suffix if Path(image_obj.id).suffix else ".png"
            new_image_name = f"{pdf_base}_img_{global_counter}{ext}"
            global_counter += 1

            # save in subfolder
            image_output_path = images_dir / new_image_name
            with open(image_output_path, "wb") as f:
                f.write(image_bytes)

            # Update markdown with wikilink: ![[nombre_imagen]]
            updated_markdown = updated_markdown.replace(
                # f"![{image_obj.id}]({image_obj.id})", f"![[{new_image_name}]]"
                f"![{image_obj.id}]({image_obj.id})",
                f"![Image](images/{new_image_name})",
            )
        updated_markdown_pages.append(updated_markdown)

    final_markdown = "\n\n".join(updated_markdown_pages)
    final_markdown = re.sub(r"\n", r"  \n", final_markdown)

    output_markdown_path = output_dir / "output.md"
    with open(output_markdown_path, "w", encoding="utf-8") as md_file:
        md_file.write(final_markdown)
    print(f"Markdown generated in {output_markdown_path}")


# Process all PDFs in INPUT_DIR
# - Important to be careful with the number of PDFs, as the Mistral API has a usage limit
#   and it could cause errors by exceeding the limit.

pdf_files = list(
    INPUT_DIR.glob("*.pdf")
)  # Get all PDFs in pdfs_to_process. So make sure to place the PDFs there.
if not pdf_files:
    print("No PDFs to process.")
    exit()

for pdf_file in pdf_files:
    process_pdf(pdf_file)
    shutil.move(str(pdf_file), DONE_DIR / pdf_file.name)
    print(f"{pdf_file.name} moved to {DONE_DIR}")
