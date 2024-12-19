import os
import requests
from bs4 import BeautifulSoup
import random
import json
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import base64
from single_file import single_file
from tqdm.auto import tqdm
from threading import Thread
import sys
sys.path.append("../")
from emd_similarity import emd_similarity
from utils import get_action_list_folder_multi_thread, get_driver, get_action_list, clean_action_list_folder


"""
Synthetic data pipeline:
For a given websight dataset html file
1. Insert fixed image url
2. Insert fixed link
3. Take screenshot
"""


def fetch_random_image_url(access_key, keyword, width, height):
    """Fetches a random image URL from Unsplash based on the keyword, width, and height. Can only process 50 requests per hour for normal api user."""
    url = f"https://api.unsplash.com/photos/random?client_id={access_key}&query={keyword}&w={width}&h={height}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        image_url = data['urls']['raw'] + f"&w={width}&h={height}"
        return image_url
    else:
        print(f"Failed to fetch image: {response.status_code}")
        return None

def update_image_urls(html_file_path, output_path, access_key_path="img_gen_key.txt"):
    """Updates all image URLs in the HTML file and saves the updated file to the output folder."""
    with open(access_key_path, 'r') as file:
        access_key = file.read().strip()
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    images = soup.find_all('img')
    for img in images:
        src = img['src']
        if "source.unsplash.com/random" in src:
            parts = src.split('/')
            size_part = parts[-2]
            keyword_part = parts[-1]
            width, height = size_part.split('x')
            keyword = keyword_part.split('?')[-1]
            
            new_image_url = fetch_random_image_url(access_key, keyword, width, height)
            if new_image_url:
                img['src'] = new_image_url
            else:
                print(f"Could not update image for {keyword}")

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))

def update_links_in_html(input_path, output_path, random_links):
    """Updates all links in the HTML file with random links from the list."""
    df = pd.read_csv(random_links, header=None)
    links = df[0].tolist()[:5000]

    with open(input_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        for a_tag in soup.find_all('a'):
            a_tag['href'] = random.choice(links)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))


def process_websight_data(source_dir, target_dir, random_links):
    """Processes all HTML files in the source directory and saves the updated files to the target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file in tqdm(os.listdir(source_dir)):
        if file.endswith('.html'):
            input_path = os.path.join(source_dir, file)
            output_path = os.path.join(target_dir, file)
            # if file exists, skip
            if os.path.exists(output_path):
                continue
            update_image_urls(input_path, output_path)
            update_links_in_html(output_path, output_path, random_links)




"""
Real-world data pipeline:
For a given url
1. Save html into a single file
2. Compare original and simplified screenshots with threshold, if different, discard
3. Simplify html
4. Take screenshot
"""

def get_single_files(url_list, target_dir, thread_num=30):
    t_list = []
    for i, url in enumerate(url_list):
        try:
            t = Thread(target=single_file, args=(url, f"{target_dir}/{i}.html"))
            t.start()
            t_list.append(t)
        except:
            pass
        if len(t_list) == thread_num:
            for t in tqdm(t_list):
                t.join()
            t_list = []

    for t in tqdm(t_list):
        t.join()


def num_of_nodes(driver, area="body", element=None):
    # number of nodes in body
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



import hashlib
import mmap

def compute_hash(image_path):
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        # Use memory-mapped file for efficient reading
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            hash_md5.update(mm)
    return hash_md5.hexdigest()

def are_different_fast(img1_path, img2_path):
    # a extremely fast algorithm to determine if two images are different,
    # only compare the size and the hash of the image
    return compute_hash(img1_path) != compute_hash(img2_path)


def simplify_graphic(driver, element, progress_bar=None, img_name={"origin": "origin.png", "after": "after.png"}):
    """utility for simplify_html, simplify the html by removing elements that are not visible in the screenshot"""
    children = element.find_elements(By.XPATH, "./*")
    deletable = True
    # check childern
    if len(children) > 0:
        for child in children:
            deletable *= simplify_graphic(driver, child, progress_bar=progress_bar, img_name=img_name)
    # check itself
    
    if deletable:
        original_html = driver.execute_script("return arguments[0].outerHTML;", element)

        driver.execute_script("""
            var element = arguments[0];
            var attrs = element.attributes;
            while(attrs.length > 0) {
                element.removeAttribute(attrs[0].name);
            }
            element.innerHTML = '';""", element)
        driver.save_full_page_screenshot(img_name["after"])
        deletable = not are_different_fast(img_name["origin"], img_name["after"])

        if not deletable:
            # be careful with children vs child_node and assining outer html to element without parent
            driver.execute_script("arguments[0].outerHTML = arguments[1];", element, original_html)
        else:
            driver.execute_script("arguments[0].innerHTML = 'MockElement!';", element)
            # set visible to false
            driver.execute_script("arguments[0].style.display = 'none';", element)
    if progress_bar:
        progress_bar.update(1)

    return deletable
            
def simplify_html(fname, save_name, pbar=True, area="html", headless=True):
    """simplify the html file and save the result to save_name, return the compression rate of the html file after simplification"""
    # copy the fname as save_name
    
    driver = get_driver(file=fname, headless=headless)
    print("driver initialized")
    original_nodes = num_of_nodes(driver, area)
    bar = tqdm(total=original_nodes) if pbar else None
    compression_rate = 1
    driver.save_full_page_screenshot(f"{fname}_origin.png")
    try:
        simplify_graphic(driver, driver.find_element(By.TAG_NAME, area), progress_bar=bar, img_name={"origin": f"{fname}_origin.png", "after": f"{fname}_after.png"})
        elements = driver.find_elements(By.XPATH, "//*[text()='MockElement!']")

        # Iterate over the elements and remove them from the DOM
        for element in elements:
            driver.execute_script("""
                var elem = arguments[0];
                elem.parentNode.removeChild(elem);
            """, element)
        
        compression_rate = num_of_nodes(driver, area) / original_nodes
        with open(save_name, "w", encoding="utf-8") as f:
            f.write(driver.execute_script("return document.documentElement.outerHTML;"))
    except Exception as e:
        print(e, fname)
    # remove images
    driver.quit()

    os.remove(f"{fname}_origin.png")
    os.remove(f"{fname}_after.png")
    return compression_rate


def are_similar(img1_path, img2_path, threshold=0.95):
    """Compares two screenshots using the Earth Mover's Distance (EMD) and returns True if the distance is below the threshold."""
    emd = emd_similarity(img1_path, img2_path, mode="RGB", max_size=64)
    print(emd)
    return emd > threshold


def process_real_html(url, output_path, headless=True):
    driver = get_driver(url=url, headless=headless)
    driver.save_full_page_screenshot(f"{output_path}_1.png")
    try:
        single_file(url, output_path)
        driver.get("file:///" + os.getcwd() + "/" + output_path)
        driver.save_full_page_screenshot(f"{output_path}_2.png")
        if not are_similar(f"{output_path}_1.png", f"{output_path}_2.png"):
            os.remove(output_path)
        else:
            simplify_html(output_path, output_path, pbar=True, headless=headless)
    except Exception as e:
        print(e, url)
    driver.quit()
    # os.remove(f"{output_path}_1.png")
    # os.remove(f"{output_path}_2.png")
    

def process_real_html_batch(url_list, target_dir, headless=True, process_num=None):
    if not process_num:
        for i, url in enumerate(url_list):
            output_path = f"{target_dir}/{i}.html"
            if os.path.exists(output_path):
                continue
            process_real_html(url, output_path, headless=headless)
    else:
        t_list = []
        for i, url in enumerate(url_list):
            output_path = f"{target_dir}/{i}.html"
            if os.path.exists(output_path):
                continue
            t = Thread(target=process_real_html, args=(url, output_path, headless))
            t.start()
            t_list.append(t)
            if len(t_list) == process_num:
                for t in t_list:
                    t.join()
                t_list = []
        for t in t_list:
            t.join()


def sample_data():
    real_data_dir = "real-world-200"
    synthetic_dir = "synthetic-300"

    stat = {}
    for files in os.listdir(real_data_dir):
        if files.endswith(".json"):
            with open(real_data_dir + "/" + files, "r") as f:
                data = json.load(f)
                stat.update({files: len(data)})

    stat = sorted(stat.items(), key=lambda x: x[1])
    lengths = [x[1] for x in stat]
    # print description of statistics of the data
    print("min:", min(lengths))
    print("max:", max(lengths))
    print("mean:", sum(lengths) / len(lengths))
    print("median:", lengths[len(lengths) // 2])
    print("10th percentile:", lengths[len(lengths) // 10])
    print("95th percentile:", lengths[len(lengths) * 95 // 100])

    print("")

    stat_syn = {}
    for files in os.listdir(synthetic_dir):
        if files.endswith(".json"):
            with open(synthetic_dir + "/" + files, "r") as f:
                data = json.load(f)
                stat_syn.update({files: len(data)})
    stat_syn = sorted(stat_syn.items(), key=lambda x: x[1])
    lengths = [x[1] for x in stat_syn]
    # print description of statistics of the data
    print("min:", min(lengths))
    print("max:", max(lengths))
    print("mean:", sum(lengths) / len(lengths))
    print("median:", lengths[len(lengths) // 2])
    print("25th percentile:", lengths[len(lengths) // 4])
    print("75th percentile:", lengths[len(lengths) * 3 // 4])

    # the 10th percentile to 98th percentile of the real data
    pre_sampled_real = [x[0] for x in stat if x[1] >= 8 and x[1] <= 400]
    # randomly sample 120 files 
    random.shuffle(pre_sampled_real)
    sampled_real = pre_sampled_real[:120]
    sampled_real = [x.replace(".json", "") for x in sampled_real]

    # the 25th percentile to 100th percentile of the synthetic data
    pre_sampled_syn = [x[0] for x in stat_syn if x[1] >= lengths[len(lengths) // 4]]
    # randomly sample 120 files
    random.shuffle(pre_sampled_syn)
    sampled_syn = pre_sampled_syn[:120]
    sampled_syn = [x.replace(".json", "") for x in sampled_syn]

    # check if the sampled files sampled_real.csv exist
    if os.path.exists("sampled_real.csv"):
        print("sampled_real.csv already exists")
        return

    # save the sampled files to a csv
    import csv
    with open("sampled_real.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["file"])
        for file in sampled_real:
            writer.writerow([file])

    with open("sampled_syn.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["file"])
        for file in sampled_syn:
            writer.writerow([file])


def get_sample_stat(file_list, folder):
    stat = []
    for files in file_list:
        files = str(files) + ".json"
        example = json.load(open(os.path.join(folder, files), encoding='utf-8'))
        if len(example) == 0:
            print(files, "empty")
            continue
        example = pd.DataFrame(example)["type"].value_counts()
        stat.append(process_value_counts(example))
    return pd.DataFrame(stat).describe()

def process_value_counts(val_count):
    val_count = val_count.to_dict()
    # have img or not, have background-image or not, have a or not
    imgs = val_count["img"] if "img" in val_count else 0
    background_imgs = val_count["background-image"] if "background-image" in val_count else 0
    a = val_count["a"] if "a" in val_count else 0
    return {
        "imgs": imgs + background_imgs,
        "a": a
    }




if __name__ == "__main__":
    
    # process synthetic data. Params: input dir, output dir, a list of url to insert into the synthetic websites
    process_websight_data("synthetic", "synthetic_processed", "url_list.csv")
        
    # process real-world data
    url_list = [
        "https://www.qpbriefing.com/2018/09/12/your-morning-briefing-1265/",
        "http://www.familyhistory.uk.com/index.php?option=com_classifieds&searchadv=&catid=88",
        "https://www.vancouver-homes-on-sale.com/blog.html?id=136133"
    ]
    # input a list of url to be processed, output saved html files of these urls. Params: list of urls, output dir, [headless webdriver], number of process
    process_real_html_batch(url_list, "test", headless=True, process_num=6)

    # get resource list and screenshots. Params: input html folder, output resource list dir, output screenshot dir, number of threads
    get_action_list_folder_multi_thread("test", "test", "test", num_threads=20)

    # clean resource list
    clean_action_list_folder("test")

    # # sample data
    # sample_data()

    # # get sample stat
    # get_sample_stat(pd.read_csv("sampled_real.csv")["file"].tolist(), "real-world-200")
    # get_sample_stat(pd.read_csv("sampled_syn.csv")["file"].tolist(), "synthetic-300")

    




    
