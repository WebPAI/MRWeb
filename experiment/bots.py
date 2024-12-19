import base64
import pandas as pd
from tqdm.auto import tqdm
from threading import Thread
import os
from PIL import Image, ImageDraw, ImageChops
import re
from openai import OpenAI
import google.generativeai as genai
import io
import json
from PIL import Image
import time
import anthropic


def encode_image(image):
    # if it is a file path
    if type(image) == str:
        try: 
            with open(image, "rb") as image_file:
                encoding = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(e)
            with open(image, "r", encoding="utf-8") as image_file:
                encoding = base64.b64encode(image_file.read()).decode('utf-8')
        return encoding
    
    else:
        # if it is a PIL image
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


class Bot:
    def __init__(self, key_path, patience=3) -> None:
        with open(key_path, "r") as f:
            self.key = f.read().replace("\n", "")
        self.patience = patience
    
    def ask(self):
        raise NotImplementedError
    
    def try_ask(self, question, image_encoding=None, verbose=False):
        for _ in range(self.patience):
            try:
                return self.ask(question, image_encoding, verbose)
            except Exception as e:
                print(e)
                time.sleep(5)
        return ""
    


class GPT4(Bot):
    def __init__(self, key_path, patience=3, model="gpt-4o") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        self.name="gpt4o"
        self.model = model
        self.file_count = 0
        
    def ask(self, question, image_encoding=None, verbose=False):
        # if the query contains an image
        if image_encoding:
            content =    {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_encoding}",
                        },
                    },
                ],
            }
        # if not
        else:
            content = {"role": "user", "content": question}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[content],
            max_tokens=4096,
            temperature=0.0,
            seed=42,
        )
        response = response.choices[0].message.content

        if verbose:
            print("####################################")
            print("question:\n", question)
            print("####################################")
            print("response:\n", response)

        return response

class Gemini(Bot):
    def __init__(self, key_path, patience=3) -> None:
        super().__init__(key_path, patience)
        GOOGLE_API_KEY= self.key
        genai.configure(api_key=GOOGLE_API_KEY, transport='rest')
        self.name = "gemini"
        self.file_count = 0
        
    def ask(self, question, image_encoding=None, verbose=False):
        generation_config ={
            "max_output_tokens": 8196,
            "temperature": 0.0,
        }
        model = genai.GenerativeModel(model_name='gemini-1.5-pro', generation_config=generation_config)

        if verbose:
            print(f"##################{self.file_count}##################")
            print("question:\n", question)

        if image_encoding:
            img = base64.b64decode(image_encoding)
            img = Image.open(io.BytesIO(img))
            response = model.generate_content([question, img]) 
        else:    
            response = model.generate_content(question)
        response.resolve()

        if verbose:
            with open(f"output_{self.file_count}_log.txt", "w", encoding="utf-8") as f:
                f.write(question + "\n\n")
                f.write(response.text + "\n\n")
        
        return response.text
    

class Claude(Bot):
    def __init__(self, key_path, patience=3) -> None:
        super().__init__(key_path, patience)
        self.client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=self.key,
        )
        self.name = "claude"
        self.file_count = 0
        
    def ask(self, question, image_encoding=None, verbose=False):

        if image_encoding:
            content =   {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_encoding,
                        },
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ],
            }
        else:
            content = {"role": "user", "content": question}


        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[content],
        )
        response = message.content[0].text
        if verbose:
            print("####################################")
            print("question:\n", question)
            print("####################################")
            print("response:\n", response)

        return response

# if __name__ == "__main__":

    # bot = Claude("../keys/claudekey.txt")
    # bot.try_ask("What is in the image?", encode_image("1.png"), verbose=True)


