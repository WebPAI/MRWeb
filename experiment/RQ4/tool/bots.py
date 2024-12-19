from openai import OpenAI
import google.generativeai as genai
from PIL import Image
import base64
import io

class Bot:
    def __init__(self, key, patience=1) -> None:
        self.key = key
        self.patience = patience
    
    def ask(self):
        raise NotImplementedError


class Gemini(Bot):
    def __init__(self, key, patience=1) -> None:
        super().__init__(key, patience)
        GOOGLE_API_KEY= self.key
        genai.configure(api_key=GOOGLE_API_KEY)
        self.name = "Gemini"
        self.file_count = 0
        
    def ask(self, question, image_encoding=None, verbose=False):
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        if verbose:
            print(f"##################{self.file_count}##################")
            print("question:\n", question)

        if image_encoding:
            img = base64.b64decode(image_encoding)
            img = Image.open(io.BytesIO(img))
            response = model.generate_content([question, img], request_options={"timeout": 3000}) 
        else:    
            response = model.generate_content(question, request_options={"timeout": 3000})
        response.resolve()

        if verbose:
            print("####################################")
            print("response:\n", response.text)
            self.file_count += 1

        return response.text


class GPT4(Bot):
    def __init__(self, key, patience=1, model="gpt-4o") -> None:
        super().__init__(key, patience)
        self.client = OpenAI(api_key=self.key)
        self.name="gpt4o"
        self.model = model
        
    def ask(self, question, image_encoding=None, verbose=False):
        
        if image_encoding:
            content =    {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_encoding}",
                },
                },
            ],
            }
        else:
            content = {"role": "user", "content": question}
        response = self.client.chat.completions.create(
        # model="gpt-4-turbo",
        model=self.model,
        messages=[
         content
        ],
        max_tokens=4096,
        )
        response = response.choices[0].message.content
        if verbose:
            print("####################################")
            print("question:\n", question)
            print("####################################")
            print("response:\n", response)

        return response