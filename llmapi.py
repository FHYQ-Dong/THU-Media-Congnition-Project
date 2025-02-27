import json
import base64
from openai import OpenAI


class LLMAPI:
    def __init__(self, apikey=None, url=None):
        self.apikey = apikey
        self.url = "https://api.siliconflow.cn/v1"
        if apikey is None:
            raise ValueError("API key is required.")
            
        self.client = OpenAI(
            api_key = self.apikey,
            base_url = self.url
        )

    def _image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _image_to_base64(self, image_bytes):
        return base64.b64encode(image_bytes).decode("utf-8")

    def analyse_images(self, prompt, image_bytes_list):
        '''
        image: bytes of image
        return: response
        '''
        return self.client.chat.completions.create(
            model = "Qwen/Qwen2-VL-72B-Instruct",
            messages = [
                {
                    "role": "system",
                    "content": "You should select some of the given images based on the prompt. Your response should be a json-like string with the pattern of {\"selected_images\": [<index1>, <index2>, ...]}. The index begins from 0."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self._image_to_base64(image_bytes=image)}",
                                "detail":"high"
                            }
                        } for image in image_bytes_list
                    ] + [{
                        "type": "text",
                        "text": prompt
                    }]
                }],
            stream = False
        )