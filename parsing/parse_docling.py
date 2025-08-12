import shutil
import os
import re
import logging
from pathlib import Path
from tqdm import tqdm

# Docling Imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# from docling.datamodel.settings import settings # Optional: if you need to tweak global Docling settings
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    ConversionResult,  # Import ConversionResult for type hinting
)


# --- Configuration ---
# Using a simple class to mimic the original 'settings' object structure
class Settings:
    # --- Directories ---
    # IMPORTANT: Adjust these paths to your actual directory structure
    BASE_DIR = Path("output")  # Or set explicitly: Path("/path/to/your/project")
    OCR_INPUT_DIR = BASE_DIR / "pdfs_to_process"
    OCR_DONE_DIR = BASE_DIR / "pdfs_done"
    OCR_OUTPUT_ROOT_DIR = BASE_DIR / "docling_output"

    # --- Docling Options ---
    # Configure Docling's PDF processing pipeline
    # See Docling documentation for all options:
    # https://kern.ai/docling-doc/datamodel/pipeline_options.html#docling.datamodel.pipeline_options.PdfPipelineOptions
    DOCLING_PDF_OPTIONS = PdfPipelineOptions(
        do_table_structure=False,  # Enable if you need table structure (might slow down)
        do_ocr=False,  # Enable OCR if PDFs might not be digitally searchable
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        images_scale=2.0,  # Adjust image resolution if needed
        generate_picture_images=True,  # CRITICAL: Set to True to extract images
    )
    DOCLING_PDF_OPTIONS.table_structure_options.mode = TableFormerMode.FAST
    DOCLING_PDF_OPTIONS.ocr_options.lang = ["es"]


# Configure basic logging for Docling and the script
logging.basicConfig(
    level=logging.INFO
)  # INFO shows Docling progress, WARNING reduces noise
log = logging.getLogger(__name__)


# --- Docling Processing Function ---


def process_pdf_docling(
    pdf_path: Path, output_dir: Path, converter: DocumentConverter, settings: Settings
):
    """
    Processes a single PDF using Docling, saves Markdown and referenced images.
    """
    output_markdown_path = output_dir / f"{pdf_path.stem}.md"
    # Docling automatically creates an image subdir named like: {pdf_path.stem}_images
    # We don't need to create it manually.

    if output_markdown_path.exists():
        log.warning(
            f"Markdown already exists for {pdf_path.name}, skipping: {output_markdown_path}"
        )
        return

    log.info(f"Processing {pdf_path.name} with Docling...")

    try:
        # Core Docling conversion step
        result: ConversionResult = converter.convert(
            str(pdf_path)
        )  # Convert expects a string path

        if not result or not result.document:
            log.error(f"Docling failed to process {pdf_path.name}. No result document.")
            return

        # Save the document as Markdown with referenced images
        # Images will be saved in a folder named '{output_markdown_path.stem}_images'
        # relative to the markdown file's location (output_dir).
        result.document.save_as_markdown(
            filename=output_markdown_path,
            image_mode="referenced",  # Saves images separately and links them
        )
        log.info(f"Markdown saved to: {output_markdown_path}")
        # Example image dir path: output_dir / f"{pdf_path.stem}_images"
        log.info(f"Images (if any) saved in subdirectory relative to Markdown file.")

        # --- Optional: Post-process Markdown for line breaks ---
        # Docling's Markdown might render single newlines as spaces in some viewers.
        # This step adds double spaces before newlines for explicit line breaks.
        # Test if you actually need this with your Markdown viewer.
        try:
            with open(output_markdown_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            # Replace single newlines (not preceded or followed by another newline)
            # with space-space-newline. Be careful not to add breaks between paragraphs.
            # This regex is a bit more careful than the original simple \n replacement.
            processed_markdown = re.sub(r"(?<!\n)\n(?!\n)", r"  \n", markdown_content)
            if processed_markdown != markdown_content:
                with open(output_markdown_path, "w", encoding="utf-8") as f:
                    f.write(processed_markdown)
                log.info(
                    f"Applied line break formatting to {output_markdown_path.name}"
                )

        except Exception as e:
            log.error(
                f"Error applying line break formatting to {output_markdown_path.name}: {e}"
            )
        # --- End Optional Post-processing ---

    except Exception as e:
        log.error(
            f"Error processing {pdf_path.name} with Docling: {e}", exc_info=True
        )  # Log traceback


# --- Main Script Logic ---


def parse_pdfs_docling(settings: Settings):
    """
    Finds PDFs in the input directory, processes them using Docling,
    and moves them to the done directory.
    """
    # Ensure directories exist
    try:
        settings.OCR_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        settings.OCR_DONE_DIR.mkdir(parents=True, exist_ok=True)
        settings.OCR_OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Error creating directories: {e}. Please check permissions.")
        return

    # Initialize Docling Converter ONCE
    # Allowed formats can be adjusted if you process other file types
    log.info("Initializing Docling DocumentConverter...")
    try:
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=settings.DOCLING_PDF_OPTIONS,
                )
            },
        )
        log.info("Docling DocumentConverter initialized.")
    except Exception as e:
        log.error(f"Failed to initialize Docling DocumentConverter: {e}", exc_info=True)
        return

    # Process all PDFs found recursively in OCR_INPUT_DIR
    pdf_files = list(settings.OCR_INPUT_DIR.rglob("*.pdf"))
    if not pdf_files:
        log.warning(f"No PDF files found in {settings.OCR_INPUT_DIR}")
        return

    log.info(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        # Determine output directory, mirroring input structure
        try:
            relative_subdir = pdf_file.relative_to(settings.OCR_INPUT_DIR).parent
            output_dir = settings.OCR_OUTPUT_ROOT_DIR / relative_subdir / pdf_file.stem
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.error(
                f"Error determining/creating output directory for {pdf_file.name}: {e}"
            )
            continue  # Skip to next file

        # Process the PDF using the dedicated function
        process_pdf_docling(pdf_file, output_dir, converter, settings)

        # Move the processed PDF to the 'done' directory, mirroring structure
        try:
            target_done_dir = settings.OCR_DONE_DIR / relative_subdir
            target_done_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pdf_file), target_done_dir / pdf_file.name)
            log.info(f"Moved {pdf_file.name} to {target_done_dir}")
        except Exception as e:
            log.error(f"Error moving {pdf_file.name} to done directory: {e}")


if __name__ == "__main__":
    # Create settings instance
    script_settings = Settings()
    # Run the main processing function
    parse_pdfs_docling(script_settings)
    log.info("Processing finished.")
