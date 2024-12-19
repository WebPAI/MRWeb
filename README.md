This is the artifact for the paper "MRWeb: An Exploration of Generating Multi-Page Resource-Aware Web Code from UI Designs". This artifact supplies the MRWeb toolkit and supplementary materials for the paper. 


This repository contains:

1. **Code implementation of MRWeb generation pipeline**, i.e., the Python script and instructions to run MRWeb to preprocess websites, and generate UI code from screenshots with resource lists. 
2.  **MRWeb Dataset**. Our experiment data (both original and generated) is available in `/data`. 
3. **Image Quality Assessment dataset**. The human annotated image pairs for image quality assessment is available upon request (yxwan9@cse.cuhk.edu.hk). 
4. **A user-friendly tool based on MRWeb**.


Quick links: [Demo video](#Demo-video) | [MRWeb Examples](#Examples) | [Code usage](#Code-usage) | [Tool usage](#MRWeb-tool) 


# Abstract

Multi-page websites dominate modern web development. However, existing design-to-code methods rely on simplified assumptions, limiting to single-page, self-contained webpages without external resource connection. To address this gap, we introduce the Multi-Page Resource-Aware Webpage (MRWeb) generation task, which transforms UI designs into multi-page, functional web UIs with internal/external navigation, image loading, and backend routing. We propose a novel resource list data structure to track resources, links, and design components. Our study applies existing methods to the MRWeb problem using a newly curated dataset of 500 websites (300 synthetic, 200 real-world). Specifically, we identify the best metric to evaluate the similarity of the web UI, assess the impact of the resource list on MRWeb generation, analyze MLLM limitations, and evaluate the effectiveness of the MRWeb tool in real-world workflows. The results show that resource lists boost navigation functionality from 0% to 66%-80% while facilitating visual similarity. Our proposed metrics and evaluation framework provide new insights into MLLM performance on MRWeb tasks. We release the MRWeb tool, dataset, and evaluation framework to promote further research



# Demo video

This video demonstrates how the MRWeb tool enables code-free development from UI designs to resource-aware, navigable websites. 



# Examples

Comparison of self-contained website and multi-page resource-aware web (MRWeb). MRWeb supports multi-page navigation, real-image loading and backend routing.

![image-20241219171847068](assets\comparison1.png)

Adding resource lists can improve the visual similarity of a generated webpage across different MLLMs and metrics since resource lists enable MLLMs to include the exact images displayed on the webpage, thus enhancing the overall similarity. Without resource lists, MLLMs can only use placeholder images in the generated web code.

![image-20241219171953522](assets\comparison2.png)


# Code usage

## 0. Setup

```shell
pip install -r requirements.txt
```



## 1. Save & Process Website

```shell
# please create an unsplash API key and input to dataset_collection/img_gen_key.txt
python dataset_collection/collect_data.py
```

## 2. MRWeb Experiment

```shell
# store your respective API keys in keys/gptkey.txt, keys/geminikey.txt, keys/claudekey.txt
python experiment/experiments.py
```

## 3. Calculate Metrics

```shell
python experiment/RQ2/metrics.py
```



# MRWeb tool

1. Start the server

```shell
cd experiment/RQ4/tool
python app.py
```

2. Visit http://127.0.0.1:5000 via local browser

3. Usage:

   1. Upload an design image.
   2. draw bounding box on the image to indicate the element, specify corresponding resources on the right.

   ![image-20241219181219420](assets\tool.png)
