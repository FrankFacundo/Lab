from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg

model_name = "starvector/starvector-1b-im2svg"

starvector = StarVectorForCausalLM.from_pretrained(model_name)

starvector.cuda()
starvector.eval()

image_pil = Image.open("assets/examples/sample-0.png")
image = starvector.process_images([image_pil])[0].cuda()
batch = {"image": image}

raw_svg = starvector.generate_im2svg(batch, max_length=1000)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)
