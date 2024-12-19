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

def get_driver(file=None, headless=True, string=None, url=None, window_size=(1920, 1080)):
    assert file or string or url, "You must provide a file or a string"
    options = Options()
    if headless:
        options.add_argument("-headless")
        driver = webdriver.Firefox(options=options)  # or use another driver
    else:
        driver = webdriver.Firefox(options=options)

    if file:
        driver.get("file:///" + os.getcwd() + "/" + file)
    elif string:
        string = base64.b64encode(string.encode('utf-8')).decode()
        driver.get("data:text/html;base64," + string)
    elif url:
        driver.get(url)

    driver.set_window_size(window_size[0], window_size[1])
    return driver

def convert_rgb_string_to_tuple(rgb_string):
    numbers = re.findall(r'\d+', rgb_string)
    return tuple(map(int, numbers))


def link_filter(s):
    idx = s.find('#')
    if idx != -1:
        return s[idx:]
    else:
        return s
    
def get_page_coordinates(element):
    # Get the bounding box of the element
    bounding_box = element.rect
    bounding_box = ((bounding_box['x'], bounding_box['y']), (bounding_box['x'] + bounding_box['width'], bounding_box['y'] + bounding_box['height']))
    return bounding_box


def get_action_list(driver, savepngpath=None, savelistpath=None, verbose=False, color=True):

    # Find all <a> elements
    links = driver.find_elements(By.TAG_NAME, 'a')

    # Dictionary to store the links and their parent elements
    links_info = []

    for link in links:
        color = link.value_of_css_property('color')
        rgb_color = convert_rgb_string_to_tuple(Color.from_string(color).rgb)
        on_click_jump_to = link.get_attribute('href')
        if on_click_jump_to is None:
            continue
        on_click_jump_to = link_filter(on_click_jump_to)

        if color:
            links_info.append({
                "position": get_page_coordinates(link),
                'type': link.tag_name,
                'color': rgb_color,
                'text': link.text,
                'on_click_jump_to': on_click_jump_to,
            })
        else:
            links_info.append({
                "position": get_page_coordinates(link),
                'type': link.tag_name,
                'text': link.text,
                'on_click_jump_to': on_click_jump_to,
            })
        # if link is empty, remove it
        if links_info[-1]['on_click_jump_to'] == "":
            links_info.pop()

    imgs = driver.find_elements(By.TAG_NAME, 'img')
    # account for the case where the image is not in img tag

    for img in imgs:
        links_info.append({
            "position": get_page_coordinates(img),
            'type': img.tag_name,
            'src': img.get_attribute('src'),
        })

    def get_background_images(driver):
        """
        Retrieves all background images from elements on the page.

        Args:
            driver (selenium.webdriver): The Selenium WebDriver instance.

        Returns:
            list: A list of dictionaries containing position, type, and src of each background image.
        """
        background_images = []
        
        # Find all elements on the page
        elements = driver.find_elements(By.CSS_SELECTOR, '*')
        
        for elem in elements:
            # Get the computed value of the 'background-image' CSS property
            bg = elem.value_of_css_property('background-image')
            
            # Continue only if a background image is set
            if bg and bg != 'none':
                # The background-image property can have multiple URLs, separated by commas
                # Example: url("image1.png"), url('image2.jpg')
                urls = re.findall(r'url\(["\']?(.*?)["\']?\)', bg)
                
                for url in urls:
                    # Append each background image's details to the list
                    background_images.append({
                        "position": get_page_coordinates(elem),  # Assumes this function is defined elsewhere
                        "type": "background-image",
                        "src": url,
                    })
        return background_images
    
    background_images = get_background_images(driver)
    links_info.extend(background_images)
    
    if verbose:
        # draw the bounding boxes on the screenshot
        driver.save_full_page_screenshot(savepngpath)
        image = Image.open(savepngpath)
        draw = ImageDraw.Draw(image)

        for info in links_info:
            if info["position"]:
                draw.rectangle(info["position"], outline="red", width=5)
        image.save(savepngpath.replace(".png", "_bounding_box.png"))

    # if savepngpath:
    #     driver.save_full_page_screenshot(savepngpath)
    
    if savelistpath:
        with open(savelistpath, 'w', encoding='utf-8') as f:
            json.dump(links_info, f, ensure_ascii=False, indent=4)

    return links_info


def get_action_list_folder(folder_path, screenshot_folder, actionlist_folder, color=False):

    file_list = os.listdir(folder_path)
    driver = get_driver(string="<html></html>")
    try:
        for file_name in tqdm(file_list):
            file_path = os.path.join(folder_path, file_name)
            
            if file_path.endswith(".html"):
                actionlist_path = os.path.join(actionlist_folder, f"{os.path.splitext(file_name)[0]}.json")
                if os.path.exists(actionlist_path):
                    continue

                driver.get("file:///" + os.getcwd() + "/" + file_path)
                screenshot_path = os.path.join(screenshot_folder, f"{os.path.splitext(file_name)[0]}.png")
                links_info = get_action_list(driver, savepngpath=screenshot_path, color=color, verbose=True)
                with open(actionlist_path, 'w', encoding='utf-8') as f:
                    json.dump(links_info, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(e, file_name)     
    driver.quit()

def get_action_list_folder_multi_thread(folder_path, screenshot_folder, actionlist_folder, color=False, num_threads=5):
    file_list = os.listdir(folder_path)
    threads = []
    drivers = [get_driver(string="<html></html>") for _ in range(num_threads)]
    try:
        for file_name in tqdm(file_list):
            file_path = os.path.join(folder_path, file_name)
            
            if file_path.endswith(".html"):
                actionlist_path = os.path.join(actionlist_folder, f"{os.path.splitext(file_name)[0]}.json")
                if os.path.exists(actionlist_path):
                    continue
                driver = drivers[len(threads)]
                driver.get("file:///" + os.getcwd() + "/" + file_path)
                screenshot_path = os.path.join(screenshot_folder, f"{os.path.splitext(file_name)[0]}.png")
                list_path = os.path.join(actionlist_folder, f"{os.path.splitext(file_name)[0]}.json")
                t = Thread(target=get_action_list, args=(driver, screenshot_path, list_path, True, True))
                t.start()
                threads.append(t)
                if len(threads) == num_threads:
                    for t in threads:
                        t.join()
                    threads = []
        for t in threads:
            t.join()
    except Exception as e:
        print(e, file_name)

    for driver in drivers:
        driver.quit()


def clean_action_list(action_list):
    original_length = len(action_list)
    # remove items with position = [ [0, 0], [0, 0] ]
    action_list = [item for item in action_list if item['position'] != [[0, 0], [0, 0]]]
    # remove items with negative position
    action_list = [item for item in action_list if item['position'][0][0] >= 0 and item['position'][0][1] >= 0 and item['position'][1][0] >= 0 and item['position'][1][1] >= 0]
    # remove items with width or height = 0
    action_list = [item for item in action_list if item['position'][1][0] - item['position'][0][0] > 0 and item['position'][1][1] - item['position'][0][1] > 0]
    # remove "a" tags with no text
    action_list = [item for item in action_list if item['type'] != 'a' or item['text'] != '']

    # print(f"compression rate: {len(action_list) / (original_length + 1e-6)}")

    return action_list

def clean_action_list_folder(actionlist_folder):
    file_list = os.listdir(actionlist_folder)
    for file_name in tqdm(file_list):
        file_path = os.path.join(actionlist_folder, file_name)
        if file_path.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                action_list = json.load(f)
            action_list = clean_action_list(action_list)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(action_list, f, ensure_ascii=False, indent=4)
