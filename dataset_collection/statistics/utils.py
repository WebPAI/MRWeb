from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import os
from PIL import Image, ImageChops, ImageDraw, ImageEnhance 
from tqdm.auto import tqdm
import time
import re
import pandas as pd
import base64
import io
from openai import OpenAI
import numpy as np
from skimage.metrics import structural_similarity as ssim
import google.generativeai as genai
import pandas as pd


def get_driver(file=None, headless=True, string=None, window_size=(1920, 1080)):
    """Initialize a webdriver, input can be the path to a html file or a string of html code"""
    assert file or string, "You must provide a file or a string"
    options = Options()
    if headless:
        options.add_argument("-headless")
        driver = webdriver.Firefox(options=options)  # or use another driver
    else:
        driver = webdriver.Firefox(options=options)

    if not string:
        driver.get("file:///" + os.getcwd() + "/" + file)
    else:
        string = base64.b64encode(string.encode('utf-8')).decode()
        driver.get("data:text/html;base64," + string)
        
    return driver


def num_of_nodes(driver, area="body", element=None):
    """Calculate number of nodes in html body, input is a webdriver"""
    element = driver.find_element(By.TAG_NAME, area) if not element else element
    script = """
    function get_number_of_nodes(base) {
        var count = 0;
        var queue = [];
        queue.push(base);
        while (queue.length > 0) {
            var node = queue.shift();
            count += 1;
            var children = node.children;
            for (var i = 0; i < children.length; i++) {
                queue.push(children[i]);
            }
        }
        return count;
    }
    return get_number_of_nodes(arguments[0]);
    """
    return driver.execute_script(script, element)

def num_of_tags(driver, area="body", element=None):
    """Calculate number of tags in the specified area, input is a webdriver"""
    element = driver.find_element(By.TAG_NAME, area) if not element else element
    script = """
    function get_number_of_tags(base) {
        var count = 0;
        var queue = [];
        queue.push(base);
        while (queue.length > 0) {
            var node = queue.shift();
            if (node.nodeType === Node.ELEMENT_NODE) {
                count += 1;
            }
            var children = node.children;
            for (var i = 0; i < children.length; i++) {
                queue.push(children[i]);
            }
        }
        return count;
    }
    return get_number_of_tags(arguments[0]);
    """
    return driver.execute_script(script, element)

def num_of_unique_tags(driver, area="body", element=None):
    """Calculate number of unique tags in the specified area, input is a webdriver"""
    element = driver.find_element(By.TAG_NAME, area) if not element else element
    script = """
    function get_number_of_unique_tags(base) {
        var uniqueTags = new Set();
        var queue = [];
        queue.push(base);
        while (queue.length > 0) {
            var node = queue.shift();
            if (node.nodeType === Node.ELEMENT_NODE) {
                uniqueTags.add(node.tagName.toLowerCase());
            }
            var children = node.children;
            for (var i = 0; i < children.length; i++) {
                queue.push(children[i]);
            }
        }
        return uniqueTags.size;
    }
    return get_number_of_unique_tags(arguments[0]);
    """
    return driver.execute_script(script, element)

def dom_tree_depth(driver, area="body", element=None):
    """Calculate the depth of the DOM tree in the specified area, input is a webdriver"""
    element = driver.find_element(By.TAG_NAME, area) if not element else element
    script = """
    function get_dom_tree_depth(node) {
        if (!node.children || node.children.length === 0) {
            return 1;
        }
        let maxDepth = 0;
        for (let i = 0; i < node.children.length; i++) {
            let childDepth = get_dom_tree_depth(node.children[i]);
            if (childDepth > maxDepth) {
                maxDepth = childDepth;
            }
        }
        return maxDepth + 1;
    }
    return get_dom_tree_depth(arguments[0]);
    """
    return driver.execute_script(script, element)