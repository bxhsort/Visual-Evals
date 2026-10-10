import base64
import requests
from io import BytesIO, StringIO
from typing import Union, Optional, Tuple, List
from PIL import Image, ImageOps
import os
from openai import OpenAI

TOKENHUB_BASE_URL = os.environ.get(
    "EVAL_API_BASE_URL",
    os.environ.get("OPENAI_BASE_URL", "https://tokenhub.sensetime.com/v1"),
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pick_next_item(current_item, item_list):
    if current_item not in item_list:
        raise ValueError("Current item is not in the list")
    current_index = item_list.index(current_item)
    next_index = (current_index + 1) % len(item_list)

    return item_list[next_index]

# Function to encode a PIL image
def encode_pil_image(pil_image):
    # Create an in-memory binary stream
    image_stream = BytesIO()
    
    # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
    pil_image.save(image_stream, format='JPEG')
    
    # Get the binary data from the stream and encode it as base64
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image


def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    """
    Load an image from a given path or URL and convert it to a PIL Image.

    Args:
        image (Union[str, Image.Image]): The image path, URL, or a PIL Image object to be loaded.
        format (str, optional): Desired color format of the resulting image. Defaults to "RGB".
        size (Optional[Tuple], optional): Desired size for resizing the image. Defaults to None.

    Returns:
        Image.Image: A PIL Image in the specified format and size.

    Raises:
        ValueError: If the provided image format is not recognized.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if (size != None):
        image = image.resize(size, Image.LANCZOS)
    return image

class GPT4v():
    def __init__(self, api_key, are_images_encoded=False, model_name="gpt-4-vision-preview"):
        """Azure OpenAI GPT-4 vision model wrapper
        Args:
            api_key (str): Azure OpenAI API key.
            are_images_encoded (bool): Whether the images are encoded in base64. Defaults to False.
        """
        self.multiple_api_keys = isinstance(api_key, list)
        self.current_key_index = 0
        self.key_lists = api_key if self.multiple_api_keys else None
        self.api_key = api_key[0] if self.multiple_api_keys else api_key
        
        if not self.api_key:
            print("API key not found.")
            exit(1)

        self.model_name = model_name
        self.use_encode = are_images_encoded
        self.base_url = TOKENHUB_BASE_URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        prompt_content = []
        text_dict = {
                    "type": "text",
                    "text": text_prompt
                }
        prompt_content.append(text_dict)

        if not isinstance(image_links, list):
            image_links = [image_links]
        original_size = None
        for image_link in image_links:
            image = load_image(image_link, size=original_size)
            original_size = image.size
            if self.use_encode == True:
                visual_dict = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(image)}"}
                    }
            else:
                visual_dict = {
                        "type": "image_url",
                        "image_url": {"url": image_link}
                    }
            prompt_content.append(visual_dict)
        return prompt_content

    def get_parsed_output(self, prompt):
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1400
        }
        try:
            response = self.client.chat.completions.create(**payload)
            return self.extract_response(response)
        except Exception as err:
            print(err)
            if self.multiple_api_keys:
                new_key = pick_next_item(self.api_key, self.key_lists)
                self.update_key(new_key)
                print("Switched to next API key.")
                response = self.client.chat.completions.create(**payload)
                return self.extract_response(response)
            return ""
    
    def extract_response(self, response):
        try:
            out = response.choices[0].message.content
            return out
        except Exception:
            print("Code is different")
            print(response)
        return ""

    def update_key(self, key):
        self.api_key = key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

class GPT4o(GPT4v):
    def __init__(self, api_key, are_images_encoded=False, model_name="gpt-4o-2024-05-13"):
        super().__init__(api_key, are_images_encoded, model_name)

if __name__ == "__main__":
    model = GPT4o(api_key="YOUR_API_KEY", model_name="gpt-4.1")
    prompt = model.prepare_prompt(['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print("prompt : \n", prompt)
    res = model.get_parsed_output(prompt)
    print("result : \n", res)
