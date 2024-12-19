from bs4 import BeautifulSoup, Comment, NavigableString
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import base64
import pandas as pd
from tqdm.auto import tqdm
from threading import Thread
import os
from PIL import Image, ImageDraw, ImageChops
import re
import shutil
import copy
from openai import OpenAI
import google.generativeai as genai
import io
import json
from PIL import Image, ImageChops
import numpy as np
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.support.color import Color
from scipy.optimize import linear_sum_assignment
import time
import logging
from transformers import GPT2Tokenizer
import csv
import math
from bots import Gemini, GPT4, Claude, encode_image


prompt_no_action = """Here is a screenshot of a web page. Please write a HTML and Tailwind CSS to make it look exactly like the original web page. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code. """


prompt_direct = """Here is a screenshot of a web page and its "action list" which specifies the links and images in the webpage. Please write a HTML and Tailwind CSS to make it look exactly like the original web page. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. The format of the action list is as follows:
    {
    "position": bounding box of format [[x1, y1], [x2, y2]], specifying the top left corner and the bottom right corner of the element;
    "type": element type;
    "color": color of the element (applies for "a" types);
    "text": element text (applies for "a" types);
    "on_click_jump_to": url of the element (the name is "src" for images);
    }
The action list is as follows: 

[ACTION_LIST]

"""

prompt_cot = """Here is a screenshot of a web page and its "action list" which specifies the links and images in the webpage. Please write a HTML and Tailwind CSS to make it look exactly like the original web page. Please think step by step, and pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. The format of the action list is as follows:
    {
    "position": bounding box of format [[x1, y1], [x2, y2]], specifying the top left corner and the bottom right corner of the element;
    "type": element type;
    "color": color of the element (applies for "a" types);
    "text": element text (applies for "a" types);
    "on_click_jump_to": url of the element (the name is "src" for images);
    }
The action list is as follows: 

[ACTION_LIST]

"""

prompt_multi = """Here is a screenshot of a web page and its "action list" which specifies the links and images in the webpage. I have an HTML file for implementing a webpage but it has some missing or wrong elements that are different from the original webpage. Please compare the two webpages and revise the original HTML implementation. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code. The format of the action list is as follows:
    {
    "position": bounding box of format [[x1, y1], [x2, y2]], specifying the top left corner and the bottom right corner of the element;
    "type": element type;
    "color": color of the element (applies for "a" types);
    "text": element text (applies for "a" types);
    "on_click_jump_to": url of the element (the name is "src" for images);
    }

The current implementation I have is: \n\n [CODE] \n\n The action list is as follows: \n\n [ACTION_LIST] \n\n"""


prompt_dict = {
    "cot": prompt_cot,
    "direct": prompt_direct,
    "no_action": prompt_no_action
}

def extract_html_content(string):
    pattern = re.compile(r'<!DOCTYPE html>.*?</html>', re.DOTALL)
    match = pattern.search(string)
    if match:
        return match.group(0)
    return "None"

def generate_code(bot, img, actionlist, prompt):
    htmlfile = bot.try_ask(prompt.replace("[ACTION_LIST]", actionlist), image_encoding=encode_image(img))
    return extract_html_content(htmlfile)

def generate_code_multi_turn(bot, img, actionlist, prompt=prompt_direct):
    htmlfile = bot.try_ask(prompt.replace("[ACTION_LIST]", actionlist), image_encoding=encode_image(img))
    htmlfile = extract_html_content(htmlfile)
    htmlfile = bot.try_ask(prompt_multi.replace("[ACTION_LIST]", actionlist).replace("[CODE]", htmlfile), image_encoding=encode_image(img))
    return extract_html_content(htmlfile)


def process_file_through_llm(bot, action_list_path, image_path, output_file_path, exp_name="direct"):
    with open(action_list_path, "r", encoding="utf-8") as file:
        actionlist = file.read()

    if exp_name != "self-refine":
        prompt = prompt_dict[exp_name]
        code = generate_code(bot, image_path, actionlist, prompt)
    else:
        prompt = prompt_direct
        code = generate_code_multi_turn(bot, image_path, actionlist, prompt)

    # ignore invalid encoding
    with open(output_file_path, 'w', encoding='utf-8', errors='ignore') as file:
        file.write(code)

    # with open(f"{output_file_path}_log.txt", "w", encoding='utf-8', errors='ignore') as file:
    #     file.write(f"{prompt}\n\n\n\n\n\n\n")
    #     file.write(f"{code}\n\n\n\n\n\n\n")

    print(f"Processed {action_list_path} and saved to {output_file_path}")


def process_files(file_list, output_folder, num_worker, model="gpt4", exp_name="direct"):
    if model == "gpt4o":
        bot = GPT4("../keys/gptkey.txt")
    elif model == "gemini":
        bot = Gemini("../keys/geminikey.txt")
    elif model == "claude":
        bot = Claude("../keys/claudekey.txt")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    threads = []
    for file_name in tqdm(file_list):
        # flush stdout
        print("", end="", flush=True)
        action_list_path, image_path = file_name + ".json", file_name + ".png"
        output_file_path = os.path.join(output_folder, os.path.basename(action_list_path).replace(".json", ".html"))
        if os.path.exists(output_file_path):
            continue
        t = Thread(target=process_file_through_llm, args=(bot, action_list_path, image_path, output_file_path, exp_name))
        t.start()
        threads.append(t)
        if len(threads) == num_worker:
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()

    print(f"Processed {len(file_list)} files and saved to {output_folder}")


if __name__ == "__main__":
    # load data
    real_list = pd.read_csv("../dataset_collection/sampled_real.csv")["file"].tolist()
    real_list = [os.path.join("../dataset_collection/all_data", str(file)) for file in real_list]
    syn_list = pd.read_csv("../dataset_collection/sampled_syn.csv")["file"].tolist()
    syn_list = [os.path.join("../dataset_collection/all_data", str(file)) for file in syn_list]
    file_list = real_list + syn_list

    models = ["gpt4o", "gemini", "claude"]
    exps = ["direct", "cot", "no_action", "self-refine"]
    for model in models:
        for exp in exps:
            output_folder = f"./results/{model}_{exp}"
            # plase modify the key file path in this function
            process_files(file_list, output_folder, 10, model=model, exp_name=exp)




    



    