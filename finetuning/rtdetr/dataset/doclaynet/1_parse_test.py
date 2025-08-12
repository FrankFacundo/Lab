import copy
import logging
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.settings import settings
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from visualization import draw_clusters

logging.basicConfig(level=logging.WARNING)

source = "/home/frank/Code/multumbabel/lab/parsing_and_chunking/pdf/SOL_UNMSM_2025_I-MH_EQUATION.pdf"

pipeline_options = PdfPipelineOptions()

pipeline_options.do_table_structure = True
pipeline_options.do_ocr = False
pipeline_options.do_code_enrichment = False
pipeline_options.do_formula_enrichment = False
pipeline_options.do_picture_classification = False

pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.ocr_options.lang = ["es"]

pipeline_options.images_scale = 4.0
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = False

settings.debug.visualize_layout = True
settings.debug.visualize_ocr = False
settings.debug.visualize_tables = False
settings.debug.visualize_cells = False
settings.debug.visualize_raw_layout = False

converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF,
    ],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,  # pipeline options go here.
        )
    },
)
conv_res = converter.convert(source)
# items = list(conv_res.document.iterate_items())
items = list(conv_res.document.iterate_items(page_no=1))

page = conv_res.pages[0]

###################
image = page.get_image(scale=pipeline_options.images_scale)
# image.show()

###################
clusters = conv_res.pages[0].predictions.layout.clusters
scale_x = page.image.width / page.size.width
scale_y = page.image.height / page.size.height
bboxes = []
labels = []
confidence = []
for c_tl in clusters:
    all_clusters = [c_tl, *c_tl.children]
    for c in all_clusters:
        x0, y0, x1, y1 = c.bbox.as_tuple()
        x0 *= scale_x
        x1 *= scale_x
        y0 *= scale_x
        y1 *= scale_y
        bboxes.append([x0, y0, x1, y1])
        labels.append(c.label.value)
        confidence.append(c.confidence)

copy_image = copy.deepcopy(page.image)
draw_clusters(copy_image, clusters, scale_x, scale_y)
copy_image.show()

###
# font = ImageFont.load_default()
# copy_image_2 = copy.deepcopy(page.image)
# draw = ImageDraw.Draw(copy_image_2, "RGBA")
# # Draw each bounding box on the image
# for idx, box in enumerate(bboxes):
#     # Draw a red rectangle with a line width of 2
#     draw.rectangle(box, outline="red", width=2)
#     draw.text(
#         (box[0], box[1]),
#         f"{labels[idx]} ({confidence[idx]:.2f})",
#         fill=(0, 0, 0, 255),  # Solid black
#         font=font,
#     )
# # Display the image with bounding boxes
# copy_image_2.show()
###

# category_id = [item[0].label.value for item in items]
page_no = page.page_no
filename = os.path.basename(source)
pathfile = source
###################
