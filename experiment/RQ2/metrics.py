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
from collections import Counter
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import sys
sys.path.append('../../')
sys.path.append('../RQ1')
from utils import get_driver, get_action_list_folder_multi_thread, clean_action_list_folder
from emd_similarity import emd_similarity
from study import mae_score, psnr_score, ssim_score, CLIPScorer, LPIPSScorer


def get_token_count(data, model_name='gpt2'):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    content = json.dumps(data)
    tokens = tokenizer.encode(content)
    return len(tokens)

def calculate_center(position):
    [[left, top], [right, bottom]] = position
    return ((left + right) / 2, (top + bottom) / 2)

def calculate_area(position):
    [[left, top], [right, bottom]] = position
    return abs((right - left) * (bottom - top))

def string_similar(s1, s2):
    s1 = set(s1.lower())
    s2 = set(s2.lower())
    overlap = len(s1.intersection(s2))
    total_chars = len(s1) + len(s2)

    if total_chars == 0:
        return 1.0
    return 2 * overlap / total_chars

def rgb_to_lab(rgb):
    """
    Convert an RGB color to Lab color space.
    RGB values should be in the range [0, 255].
    """
    # Create an sRGBColor object from RGB values
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    
    # Convert to Lab color space
    lab_color = convert_color(rgb_color, LabColor)
    
    return lab_color.get_value_tuple()

def ciede2000(rgb1, rgb2):

    # Convert RGB to Lab
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)

    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Calculate C and h for both colors
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    
    # Calculate the mean C
    C_bar = (C1 + C2) / 2
    
    # Calculate G
    G = 0.5 * (1 - math.sqrt((C_bar ** 7) / (C_bar ** 7 + 25 ** 7)))

    # Adjusted a values
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    # Recalculate C with adjusted a values
    C1_prime = math.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = math.sqrt(a2_prime ** 2 + b2 ** 2)
    
    # Calculate h values
    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360

    # Calculate delta L, delta C, and delta H
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    h_diff = h2_prime - h1_prime
    if abs(h_diff) > 180:
        if h2_prime <= h1_prime:
            h2_prime += 360
        else:
            h1_prime += 360

    delta_h_prime = h2_prime - h1_prime
    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime) / 2)

    # Calculate the mean L and mean H
    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    H_bar_prime = (h1_prime + h2_prime) / 2
    if abs(h1_prime - h2_prime) > 180:
        H_bar_prime += 180

    # Calculate T
    T = 1 - 0.17 * math.cos(math.radians(H_bar_prime - 30)) + \
        0.24 * math.cos(math.radians(2 * H_bar_prime)) + \
        0.32 * math.cos(math.radians(3 * H_bar_prime + 6)) - \
        0.20 * math.cos(math.radians(4 * H_bar_prime - 63))

    # Calculate delta_theta, R_C, and S_L, S_C, S_H
    delta_theta = 30 * math.exp(-(((H_bar_prime - 275) / 25) ** 2))
    R_C = 2 * math.sqrt((C_bar_prime ** 7) / (C_bar_prime ** 7 + 25 ** 7))
    S_L = 1 + ((0.015 * ((L_bar_prime - 50) ** 2)) / math.sqrt(20 + ((L_bar_prime - 50) ** 2)))
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T

    # Calculate R_T
    R_T = -math.sin(math.radians(2 * delta_theta)) * R_C

    # Calculate delta E
    delta_E = math.sqrt(
        (delta_L_prime / S_L) ** 2 +
        (delta_C_prime / S_C) ** 2 +
        (delta_H_prime / S_H) ** 2 +
        R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )
    
    # for visualization purpose, we consider the delta_E value larger than 100 as completely different
    return max(0, 1 - (delta_E / 100))

def calculate_metrics(json1, json2, driver1):
    scroll_width = driver1.execute_script("return document.documentElement.scrollWidth")
    scroll_height = driver1.execute_script("return document.documentElement.scrollHeight")

    items_info_1 = []
    items_info_2 = []

    # Combine all relevant items (links and images)
    for item in json1:
        if item['type'] in ['a', 'img', 'background-image']:
            items_info_1.append(item)

    for item in json2:
        if item['type'] in ['a', 'img', 'background-image']:
            items_info_2.append(item)

    if len(items_info_1) == 0:
        raise ValueError("No valid items found in JSON1")

    matches = []
    items_info_1_copy = items_info_1[:]
    items_info_2_copy = items_info_2[:]

    for item1 in items_info_1_copy:
        for item2 in items_info_2_copy:
            if item1['type'] == 'a' and item2['type'] == 'a' and item1.get('on_click_jump_to') == item2.get('on_click_jump_to'):
                matches.append((item1, item2))
                items_info_2_copy.remove(item2)
                break
            elif item1['type'] in ['img', 'background-image'] and item2['type'] in ['img', 'background-image'] and item1.get('src') == item2.get('src'):
                matches.append((item1, item2))
                items_info_2_copy.remove(item2)
                break

    if len(matches) == 0:
        return {
            "match_ratio": 0,
            "average_center_offset": np.nan,
            "average_area_difference": np.nan,
            "json_file_token": get_token_count(json1),
            "average_text_similarity": np.nan,
            "average_color_difference": np.nan
        }

    match_ratio = len(matches) / len(items_info_1)
    center_offsets = []
    area_differences = []
    text_similarities = []
    color_differences = []

    for item1, item2 in matches:
        # Calculate center offsets
        center1 = calculate_center(item1['position'])
        center2 = calculate_center(item2['position'])
        center_offset_x = abs(center1[0] - center2[0]) / scroll_width
        center_offset_y = abs(center1[1] - center2[1]) / scroll_height
        center_offset = 1 - max(center_offset_x, center_offset_y)
        if center_offset < 0:
            center_offset = 0
        center_offsets.append(center_offset)

        # Calculate area differences
        area1 = calculate_area(item1['position'])
        area2 = calculate_area(item2['position'])
        area_difference = 1 - min(abs(area1 - area2) / (area1), 1)
        if area_difference < 0:
            area_difference = 0
        area_differences.append(area_difference)

        # Calculate text similarity for links only
        if item1['type'] == 'a' and item2['type'] == 'a':
            text_similarity = string_similar(item1['text'], item2['text'])
            text_similarities.append(text_similarity)

        # Calculate color differences for links only
        if item1['type'] == 'a' and item2['type'] == 'a':
            color_difference = ciede2000(item1['color'], item2['color'])
            color_differences.append(color_difference)

    average_center_offset = sum(center_offsets) / len(center_offsets)
    average_area_difference = sum(area_differences) / len(area_differences)
    average_text_similarity = sum(text_similarities) / len(text_similarities) if text_similarities else None
    average_color_difference = sum(color_differences) / len(color_differences) if color_differences else None
    json_file_token = get_token_count(json1)

    result = {
        "match_ratio": match_ratio,
        "average_center_offset": average_center_offset,
        "average_area_difference": average_area_difference,
        "json_file_token": json_file_token
    }

    if average_text_similarity is not None:
        result["average_text_similarity"] = average_text_similarity
    if average_color_difference is not None:
        result["average_color_difference"] = average_color_difference

    return result

def fine_grained_performance(ori_dir, gen_dir, output_path):
    
    # if exist, load the existing data
    if os.path.exists(output_path):
        result = pd.read_csv(output_path)
    else:
        data = {}
        data['file_name'] = []
        data['match_ratio'] = []
        data['average_center_offset'] = []
        data['average_area_difference'] = []
        data['average_text_similarity'] = []
        data['average_color_difference'] = []
        data['json_file_token'] = []
        result = pd.DataFrame(data)

    gen_files = os.listdir(gen_dir)
    driver = get_driver(string="<html></html>")

    for file_name in tqdm(gen_files):
        if file_name.endswith('.html'):
            file_name = file_name.split('.')[0]

            ori_path = os.path.join(ori_dir, file_name + '.json')
            gen_path = os.path.join(gen_dir, file_name + '.json')

            if gen_path in result['file_name'].values:
                continue

            with open(ori_path, 'r', encoding='utf-8') as f:
                ori_json = json.load(f)
            with open(gen_path, 'r', encoding='utf-8') as f:
                gen_json = json.load(f)

            try:
                driver.get("file:///" + os.getcwd() + "/" + ori_path.replace('.json', '.html'))
                result_dict = calculate_metrics(ori_json, gen_json, driver)
                result_dict['file_name'] = gen_path
                result = pd.concat([result, pd.DataFrame([result_dict])], ignore_index=True)
                if len(result) % 10 == 0:
                    result.to_csv(output_path, index=False)
            except Exception as e:
                print(e, file_name)

    driver.quit()
    result.to_csv(output_path, index=False)


def calulate_similarity(img1_path, img2_path):
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    return {
        "NEMD": emd_similarity(img1_path, img2_path),
        "MAE": mae_score(img1, img2)
    }



def visual_similarity(ori_dir, gen_dir, output_path):
    # if exist, load the existing data
    if os.path.exists(output_path):
        result = pd.read_csv(output_path)
    else:
        data = {}
        data['file_name'] = []
        data['NEMD'] = []
        data['MAE'] = []
        result = pd.DataFrame(data)

    gen_files = os.listdir(gen_dir)

    for file_name in tqdm(gen_files):
        if file_name.endswith('.html'):
            file_name = file_name.split('.')[0]
            ori_path = os.path.join(ori_dir, file_name + '.png')
            gen_path = os.path.join(gen_dir, file_name + '.png')

            if gen_path in result['file_name'].values:
                continue

            try:
                result_dict = calulate_similarity(ori_path, gen_path)
                result_dict['file_name'] = gen_path
                result = pd.concat([result, pd.DataFrame([result_dict])], ignore_index=True)
                if len(result) % 10 == 0:
                    result.to_csv(output_path, index=False)
            except Exception as e:
                print(e, file_name)

    result.to_csv(output_path, index=False)


def visual_similarity_update(ori_dir, output_path):
    # if exist, load the existing data
    if os.path.exists(output_path):
        result = pd.read_csv(output_path)
        if 'PSNR' not in result.columns:
            result['PSNR'] = ""
        if 'SSIM' not in result.columns:
            result['SSIM'] = ""
        if 'CLIP' not in result.columns:
            result['CLIP'] = ""
        if 'LPIPS' not in result.columns:
            result['LPIPS'] = ""
    clip_scorer = CLIPScorer()
    lpips_scorer = LPIPSScorer()
    for i in tqdm(range(len(result))):
        try:
            if str(result.loc[i, 'PSNR']) != "nan":
                continue
            file_name = result.loc[i, 'file_name'].split('/')[-1]
            ori_path = os.path.join(ori_dir, file_name)
            img1 = Image.open(ori_path).convert('RGB')
            img2 = Image.open(result.loc[i, 'file_name']).convert('RGB')
            result.loc[i, 'PSNR'] = psnr_score(img1, img2)
            result.loc[i, 'SSIM'] = ssim_score(img1, img2)
            result.loc[i, 'CLIP'] = clip_scorer.score(img1, img2)
            result.loc[i, 'LPIPS'] = lpips_scorer.score(img1, img2)
            if i % 10 == 0:
                result.to_csv(output_path, index=False)
        except Exception as e:
            print(e, file_name)

    result.to_csv(output_path, index=False)

def fine_grained_performance_update(ori_dir, output_path):
    # if exist, load the existing data
    if os.path.exists(output_path):
        result = pd.read_csv(output_path)
        if 'action_list_len' not in result.columns:
            result['action_list_len'] = ""
        if 'image_size' not in result.columns:
            result['image_size'] = ""
    for i in tqdm(range(len(result))):
        try:
            if not str(result.loc[i, 'action_list_len']) in ["nan", ""] and not str(result.loc[i, 'image_size']) in ["nan", ""]:
                continue
            file_name = result.loc[i, 'file_name'].split('/')[-1]
            ori_path = os.path.join(ori_dir, file_name)
            with open(ori_path, 'r', encoding='utf-8') as f:
                ori_json = json.load(f)
            result.loc[i, 'action_list_len'] = len(ori_json)
            img_path = ori_path.replace('.json', '.png')
            img = Image.open(img_path).convert('RGB')
            result.loc[i, 'image_size'] = img.size[0] * img.size[1]
            if i % 10 == 0:
                result.to_csv(output_path, index=False)
        except Exception as e:
            print(e, file_name)

    result.to_csv(output_path, index=False)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    ori_dir = "../../dataset_collection/all_data"
    result_dir = "../results/"
    output_path = "./performance_finegrain_updated.csv"

    ### get action list and take screenshots for all generated html files

    for file in tqdm(os.listdir(result_dir)):
        if not os.path.isdir(os.path.join(result_dir, file)):
            continue
        gen_dir = os.path.join(result_dir, file)
        get_action_list_folder_multi_thread(gen_dir, gen_dir, gen_dir, color=True, num_threads=10)
        clean_action_list_folder(gen_dir)


    ### calculate the performance metrics for all generated html files

    for file in tqdm(os.listdir(result_dir)):
        if not os.path.isdir(os.path.join(result_dir, file)):
            continue
        gen_dir = os.path.join(result_dir, file)
        fine_grained_performance(ori_dir, gen_dir, output_path)
        visual_similarity(ori_dir, gen_dir, output_path)

    ### update the visual similarity metrics
    
    # visual_similarity_update(ori_dir, output_path)
    # fine_grained_performance_update(ori_dir, output_path)
    



    