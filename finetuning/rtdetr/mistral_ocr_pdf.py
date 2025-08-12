import base64

from mistralai import Mistral

api_key = ""

client = Mistral(api_key=api_key)


ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        # "document_url": signed_url.url,
        "document_url": "",
    },
    include_image_base64=True,
)

# print(ocr_response)


def decode_base64_image(encoded_str, output_file):
    # Decode the base64 string into binary data
    image_data = base64.b64decode(encoded_str)

    # Write the binary data to an output file
    with open(output_file, "wb") as file:
        file.write(image_data)
    print(f"Image saved as {output_file}")


# Example usage:
# Replace 'your_base64_encoded_string_here' with your actual base64 string.
page_nb = 46

for image_nb in range(len(ocr_response.pages[page_nb].images)):
    base64_string = ocr_response.pages[page_nb].images[image_nb].image_base64
    decode_base64_image(
        base64_string[len("data:image/jpeg;base64,") :],
        ocr_response.pages[page_nb].images[image_nb].id,
    )

# print(ocr_response.pages[page_nb].markdown)

with open("test.md", "w") as file:
    file.write(ocr_response.pages[page_nb].markdown)
