import sys
sys.path.append("../../")
from utils import get_driver
from emd_similarity import emd_similarity, process_imgs
import os
from tqdm import tqdm
import pandas as pd
import time
import random
import sqlite3
import open_clip
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
import itertools
import pandas as pd


def take_screenshot_for_dir(dir):
    # print(f"getting driver")
    driver = get_driver(string="<html></html>", headless=True)
    for file in tqdm(os.listdir(dir)):
        if file.endswith(".html"):
            if os.path.exists(os.path.join(dir, file.replace(".html", ".png"))):
                continue
            driver.get(f"file://{os.path.join(os.path.abspath(dir), file)}")
            time.sleep(2)
            driver.save_full_page_screenshot(os.path.join(dir, file.replace(".html", ".png")))
    driver.quit()
    return []

def process_results(func):
    """process every file in every subdirectory in results/ with a function"""
    result = []
    for dir in tqdm(os.listdir("../results/")):
        dir = os.path.join("../results/", dir)
        if os.path.isdir(dir):
            result += func(dir)
    return result

def match_screenshots_with_original(dir, original_dir = "../../dataset_collection/all_data/"):
    """match screenshots of generated html with original screenshots to form [original, generated] pairs"""
    result = []
    for file in os.listdir(dir):
        if file.endswith(".png"):
            abs_path_gen = os.path.join(os.path.abspath(dir), file)
            abs_path_orig = os.path.join(os.path.abspath(original_dir), file)
            result.append((abs_path_orig, abs_path_gen))
    return result

def copy_pngs_to_annotation_tool(result):
    """copy pngs in results to annotation_tool/static/images/[dir_name]/[png_file] for annotation"""
    os.makedirs(f"annotation_tool/static/images/original", exist_ok=True)
    for original_png, generated_png in tqdm(result):
        if os.name == "posix":
            dir_name = generated_png.split("results/")[1].split("/")[0]
            png_name = generated_png.split("/")[-1]
        else:
            dir_name = generated_png.split("results\\")[1].split("\\")[0]
            png_name = generated_png.split("\\")[-1]
        os.makedirs(f"annotation_tool/static/images/{dir_name}", exist_ok=True)
        # if linux or mac, use cp command
        if os.name == "posix":
            os.system(f"cp {original_png} annotation_tool/static/images/original/{png_name}")
            os.system(f"cp {generated_png} annotation_tool/static/images/{dir_name}/{png_name}")
        # if windows, use copy command and replace / with \ and set overwrite flag to true
        else:
            os.system(f"copy {original_png} annotation_tool\\static\\images\\original\\{png_name} /Y")
            os.system(f"copy {generated_png} annotation_tool\\static\\images\\{dir_name}\\{png_name} /Y")
        
def initialize_annotation_tool(sample_size=600):
    result = process_results(match_screenshots_with_original)
    random.shuffle(result)
    result = result[:sample_size]
    pd.DataFrame(result, columns=["original", "generated"]).to_csv("original_generated_pairs.csv", index=False)

    # remove old pngs
    os.system("rm -rf annotation_tool/static/images")
    # copy pngs to annotation_tool/static/images/[dir_name]/[png_file] for annotation
    copy_pngs_to_annotation_tool(result)

    # generate new csv file for annotation_tool with new paths
    new_result = []
    for original_png, generated_png in result:
        if os.name == "posix":
            dir_name = generated_png.split("results/")[1].split("/")[0]
            png_name = generated_png.split("/")[-1]
            new_result.append((f"images/original/{png_name}", f"images/{dir_name}/{png_name}"))
        else:
            dir_name = generated_png.split("results\\")[1].split("\\")[0]
            png_name = generated_png.split("\\")[-1]
            new_result.append((f"images\\original\\{png_name}", f"images\\{dir_name}\\{png_name}"))
    pd.DataFrame(new_result, columns=["original", "generated"]).to_csv("annotation_tool/image_pairs.csv", index=False)
    os.remove("./annotation_tool/instance/annotation_tool.db")


def annotation_to_table(db_path="annotation_tool/instance/annotation_tool.db"):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Load the data from each table
    image_pairs = pd.read_sql_query("SELECT * FROM image_pair;", conn)
    annotations = pd.read_sql_query("SELECT * FROM annotation;", conn)
    users = pd.read_sql_query("SELECT * FROM user;", conn)

    # select user ids than have rated all image pairs
    user_ids = annotations.groupby("user_id")["image_pair_id"].count()
    user_ids = user_ids[user_ids < 500].index
    annotations = annotations[~annotations["user_id"].isin(user_ids)]
    
    # Close the connection
    conn.close()

    # Merge image pairs with annotations on the image pair ID
    combined_data = pd.merge(annotations, image_pairs, left_on='image_pair_id', right_on='id')
    
    # Pivot the data to get each user's rating in separate columns
    # This assumes 'user_id' identifies each user uniquely in the annotations table
    combined_data_pivot = combined_data.pivot(index='image_pair_id', columns='user_id', values='rating')

    # Rename the columns to make it clear each column is a rating by a specific user
    combined_data_pivot.columns = [f'user_{user_id}_rating' for user_id in combined_data_pivot.columns]

    # Reset index to make 'image_pair_id' a column instead of an index
    combined_data_pivot.reset_index(inplace=True)

    # Merge image pair details if needed (e.g., image filenames)
    combined_table = pd.merge(image_pairs[['id', 'original_path', 'generated_path']], combined_data_pivot, left_on='id', right_on='image_pair_id')
    combined_table.drop(columns=['id'], inplace=True)
    
    return combined_table


class CLIPScorer:
    def __init__(self, model_name='ViT-B-32-quickgelu', pretrained='openai'):
        """
        Initializes the CLIPScorer with the specified model.

        Args:
            model_name (str): The name of the CLIP model to use.
            pretrained (str): Specifies whether to load pre-trained weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)

    def score(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculates the CLIP score (cosine similarity) between two images.

        Args:
            img1 (Image.Image): The first image as a PIL Image.
            img2 (Image.Image): The second image as a PIL Image.

        Returns:
            float: The cosine similarity score between the two images.
        """
        # Preprocess the images
        image1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        image2 = self.preprocess(img2).unsqueeze(0).to(self.device)

        # Get the image features from CLIP using openclip
        with torch.no_grad():
            image1_features = self.model.encode_image(image1)
            image2_features = self.model.encode_image(image2)

        # Normalize the features to unit length
        image1_features /= image1_features.norm(dim=-1, keepdim=True)
        image2_features /= image2_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity between the two image features
        cosine_similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features)
        return cosine_similarity.item()


class LPIPSScorer:
    def __init__(self, net='vgg'):
        """
        Initializes the LPIPS scorer with the specified network type.

        Args:
            net (str): The network to use for LPIPS calculation ('vgg', 'alex', or 'squeeze').
        """
        self.loss_fn = lpips.LPIPS(net=net)  # Load the LPIPS model
        self.device = "cuda" if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        self.loss_fn.to(self.device)

    def score(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Calculates the LPIPS similarity score between two images.

        Args:
            image1 (Image.Image): The first image as a PIL Image.
            image2 (Image.Image): The second image as a PIL Image.

        Returns:
            float: The LPIPS similarity score between the two images.
        """
        image1, image2 = process_imgs(image1, image2, 512)
        # 
        # Convert images to float tensors
        transform = transforms.Compose([
            transforms.ToTensor(),          # Convert to tensor
            transforms.Lambda(lambda x: x.to(dtype=torch.float32)),  # Convert to float32
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        img1_tensor = transform(image1).unsqueeze(0).to(self.device)
        img2_tensor = transform(image2).unsqueeze(0).to(self.device)

        # Calculate the LPIPS similarity score
        with torch.no_grad():  # Disable gradients for inference
            lpips_score = self.loss_fn(img1_tensor, img2_tensor)

        return lpips_score.item()


def ssim_score(img1, img2):
    # resize images to match the size of the smaller image
    img1, img2 = process_imgs(img1, img2, 512)
    return ssim(img1, img2, channel_axis=-1, data_range=255)

def psnr_score(img1, img2):
    """peak signal-to-noise ratio, it is a pixel-based metric"""
    img1, img2 = process_imgs(img1, img2, 512)
    return psnr(img1, img2)

def mae_score(img1, img2):
    """mean absolute error, it is a pixel-based metric"""
    img1, img2 = process_imgs(img1, img2, 512)
    # max_mae = np.mean(np.maximum(img1, 255 - img1))
    mae = np.mean(np.abs(img1 - img2))
    # return {"mae": mae, "normalized_mae": 1 - mae / max_mae}
    return mae


def evaluate_metrics(image_pairs="./metrics.csv", output_path="metrics.csv"):
    """evaluate automatic metrics for each image pair in the csv file"""
    image_pairs = pd.read_csv(image_pairs)
    image_pairs["mae"] = "" if "mae" not in image_pairs.columns else image_pairs["mae"]
    image_pairs["mae_normalize"] = "" if "mae_normalize" not in image_pairs.columns else image_pairs["mae_normalize"]
    image_pairs["emd"] = "" if "emd" not in image_pairs.columns else image_pairs["emd"]
    image_pairs["emd_normalized"] = "" if "emd_normalized" not in image_pairs.columns else image_pairs["emd_normalized"]
    image_pairs["psnr"] = "" if "psnr" not in image_pairs.columns else image_pairs["psnr"]
    image_pairs["ssim"] = "" if "ssim" not in image_pairs.columns else image_pairs["ssim"]
    image_pairs["clip"] = "" if "clip" not in image_pairs.columns else image_pairs["clip"]
    image_pairs["lpips"] = "" if "lpips" not in image_pairs.columns else image_pairs["lpips"]
    with tqdm(total=len(image_pairs)) as pbar:
        for i, row in image_pairs.iterrows():
            if row["lpips"] != "":
                pbar.update(1)
                continue
            img1 = Image.open(f"annotation_tool/static/{row['original']}").convert("RGB")
            img2 = Image.open(f"annotation_tool/static/{row['generated']}").convert("RGB")
            mae = mae_score(img1, img2)
            emd = emd_similarity(img1, img2)
            image_pairs.at[i, "mae"] = mae["mae"]
            image_pairs.at[i, "mae_normalize"] = mae["normalized_mae"]
            image_pairs.at[i, "emd"] = emd["cost"]
            image_pairs.at[i, "emd_normalized"] = emd["normalized_sim"]
            image_pairs.at[i, "psnr"] = psnr_score(img1, img2)
            image_pairs.at[i, "ssim"] = ssim_score(img1, img2)
            image_pairs.at[i, "clip"] = CLIPScorer().score(img1, img2)
            image_pairs.at[i, "lpips"] = LPIPSScorer().score(img1, img2)
            pbar.update(1)

            if i % 10 == 0:
                image_pairs.to_csv(output_path, index=False)

    image_pairs.to_csv(output_path, index=False)


def normalize_annotation(annotation="annotation.csv"):
    """Normalize the annotation data and resample to ensure uniform distribution of ratings"""
    annotation = pd.read_csv(annotation)
    user_rating_columns = [col for col in annotation.columns if col.startswith('user_') and col.endswith('_rating')]
    # Normalize each user rating column to a z-score
    for col in user_rating_columns:
        mean = annotation[col].mean()
        std = annotation[col].std()
        annotation[col] = (annotation[col] - mean) / std


    # Function to remove outliers from a row using IQR
    def remove_outliers(row):
        q1 = row.quantile(0.25)
        q3 = row.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return row[(row >= lower_bound) & (row <= upper_bound)]

    # Remove outliers for each row and calculate the average and variance
    annotation["rating_avg"] = annotation[user_rating_columns].apply(remove_outliers, axis=1).mean(axis=1)
    annotation["rating_var"] = annotation[user_rating_columns].apply(remove_outliers, axis=1).var(axis=1)

    # resample the data so that the resulting distribution is approximately uniform
    avg = annotation["rating_avg"]
    bin_edges = np.linspace(avg.min(), avg.max(), num=4)  # Split into 3 bins
    annotation['bin'] = pd.cut(avg, bins=bin_edges, include_lowest=True)

    # Determine target sample size per bin
    target_sample_size = min(annotation['bin'].value_counts())  # Equalize to smallest bin size

    # Sample uniformly from each bin
    uniform_sampled_data = annotation.groupby('bin', observed=False).apply(
        lambda x: x.sample(n=target_sample_size, random_state=42), include_groups=False
    )


    uniform_sampled_data.reset_index(drop=True).to_csv("normalized_annotation.csv", index=False)

    # Save data for each bin
    for i, (bin_label, bin_data) in enumerate(uniform_sampled_data.reset_index().groupby('bin', observed=False)):
        # Create a filename based on the bin label
        bin_filename = f"normalize_annotation_bin_{i}.csv"
        bin_data.drop(columns=['bin', "level_1"], inplace=True) # Drop the 'bin' column before saving
        bin_data.to_csv(bin_filename, index=False)
        print(f"Saved normalized data for bin {str(bin_label)} to {i}")


def merge_result(annotation="normalized_annotation.csv", metrics="metrics.csv", output="merged.csv"):
    annotation = pd.read_csv(annotation)
    metrics = pd.read_csv(metrics)
    # Merging the two DataFrames on 'original' and 'generated' columns, as these correspond to 'original_path' and 'generated_path' in the annotation DataFrame
    merged_df = pd.merge(
        annotation,
        metrics,
        left_on=['original_path', 'generated_path'],
        right_on=['original', 'generated'],
        how='inner'
    )

    # Dropping the redundant 'original' and 'generated' columns after merge
    user_rating_columns = [col for col in merged_df.columns if col.startswith('user_') and col.endswith('_rating')]
    merged_df.drop(columns=['original', 'generated', 'image_pair_id'], inplace=True)
    merged_df.drop(columns=user_rating_columns, inplace=True)
    merged_df.to_csv(output, index=False)


def calculate_correspondance_metrics(objective_scores, subjective_scores, variances):
    # 1. Metric 1: Correlation Coefficient after Variance-Weighted Regression
    # Compute weights as inverse of variance
    weights = 1 / variances

    # Perform weighted linear regression using numpy
    X = np.vstack([objective_scores, np.ones(len(objective_scores))]).T
    W = np.diag(weights)
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ subjective_scores)
    predicted_weighted = X @ beta

    # Calculate correlation coefficient after variance-weighted regression
    correlation_weighted = np.corrcoef(subjective_scores, predicted_weighted)[0, 1]

    # Outlier ratio for weighted regression
    residuals_weighted = subjective_scores - predicted_weighted
    std_dev_weighted = np.std(residuals_weighted)
    outlier_count_weighted = np.sum(np.abs(residuals_weighted) > 2 * std_dev_weighted)
    outlier_ratio_weighted = outlier_count_weighted / len(residuals_weighted)

    # 2. Metric 2: Correlation Coefficient after Nonlinear Regression (Logistic Mapping)

    # Define the logistic function for fitting
    def logistic_function(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Fit logistic curve to data
    popt, _ = curve_fit(logistic_function, objective_scores, subjective_scores, p0=[1, 1/np.std(objective_scores), np.median(objective_scores)], maxfev=5000)
    predicted_logistic = logistic_function(objective_scores, *popt)

    # Calculate correlation coefficient after nonlinear regression
    correlation_nonlinear = np.corrcoef(subjective_scores, predicted_logistic)[0, 1]

    # 3. Metric 3: Spearman Rank-Order Correlation Coefficient
    spearman_corr, _ = spearmanr(objective_scores, subjective_scores)

    # 4. Metric 4: Outlier Ratio (±2 standard deviations after nonlinear mapping)
    residuals = subjective_scores - predicted_logistic
    std_dev = np.std(residuals)
    outlier_count = np.sum(np.abs(residuals) > 2 * std_dev)
    outlier_ratio = outlier_count / len(residuals)

    # 5. Additional Metric: Mean Absolute Error (MAE) after Nonlinear Regression
    mae = np.mean(np.abs(subjective_scores - predicted_logistic))
    mae_weighted = np.mean(np.abs(subjective_scores - predicted_weighted) * weights)

    # 6. Additional Metric: Root Mean Square Error (RMS) after Nonlinear Regression
    rms = np.sqrt(np.mean((subjective_scores - predicted_logistic) ** 2))
    rms_weighted = np.sqrt(np.mean((subjective_scores - predicted_weighted) ** 2 * weights))

    # Display the results
    metrics = {
        'CC (Weighted Regression)': correlation_weighted,
        'MAE (Weighted Regression)': mae_weighted,
        'RMS (Weighted Regression)': rms_weighted,
        'OR (Weighted Regression)': outlier_ratio_weighted,
        # 'STD (Weighted Regression)': std_dev_weighted,
        'CC (Nonlinear Regression)': correlation_nonlinear,
        'MAE (Nonlinear Regression)': mae,
        'RMS (Nonlinear Regression)': rms,
        'OR (Nonlinear Regression)': outlier_ratio,
        # 'STD (Nonlinear Regression)': std_dev,
        'Rank-Order': abs(spearman_corr),
    }
    return metrics

def get_correspondance_metrics(merged="merged.csv"):
    merged = pd.read_csv(merged)
    subjective_scores = merged["rating_avg"]
    variances = merged["rating_var"]
    results = {
        'CC (Weighted Regression)': [],
        'MAE (Weighted Regression)': [],
        'RMS (Weighted Regression)': [],
        'OR (Weighted Regression)': [],
        # 'STD (Weighted Regression)': [],
        'CC (Nonlinear Regression)': [],
        'MAE (Nonlinear Regression)': [],
        'WRMS (Nonlinear Regression)': [],
        'OR (Nonlinear Regression)': [],
        # 'STD (Nonlinear Regression)': [],
        'Rank-Order': [],
    }
    metric_names = []

    for col in tqdm(merged.columns):
        if col in ["rating_avg", "rating_var", "original_path", "generated_path"]:
            continue
        # prevent overflow or inf values
        if col == "mae":
            objective_scores = merged[col]/(255)
        if col == "emd":
            objective_scores = merged[col]/(128*128*3*255)
        elif col == "psnr":
            objective_scores = merged[col].replace(np.inf, 2e2)
        else:
            objective_scores = merged[col]
        metrics = calculate_correspondance_metrics(objective_scores, subjective_scores, variances)
        metric_names.append(col)
        for key in metrics:
            results[key].append(metrics[key])

    return pd.DataFrame(results, index=metric_names)

def get_annotation_validity(annotation="normalized_annotation.csv"):
    annotation_z = pd.read_csv(annotation)
    user_rating_columns = [col for col in annotation_z.columns if col.startswith('user_') and col.endswith('_rating')]
    
    # Get all possible pairs of user rating columns
    pairs = list(itertools.combinations(user_rating_columns, 2))
    
    # Calculate correlation coefficient for each pair
    correlation_coefficients = []
    rank_order_correlations = []
    for col1, col2 in pairs:
        corr = annotation_z[col1].corr(annotation_z[col2])
        rank_order_corr, _ = spearmanr(annotation_z[col1], annotation_z[col2])
        if not np.isnan(corr):
            correlation_coefficients.append(corr)
        if not np.isnan(rank_order_corr):
            rank_order_correlations.append(rank_order_corr)
    
    # Calculate the average correlation coefficient
    if correlation_coefficients:
        average_correlation = sum(correlation_coefficients) / len(correlation_coefficients)
    else:
        average_correlation = None

    if rank_order_correlations:
        average_rank_order = sum(rank_order_correlations) / len(rank_order_correlations)
    else:
        average_rank_order = None

    description = {
        'Average Correlation Coefficient': average_correlation,
        'Average Rank-Order Correlation Coefficient': average_rank_order,
        'Max Correlation Coefficient': max(correlation_coefficients) if correlation_coefficients else None,
        'Min Correlation Coefficient': min(correlation_coefficients) if correlation_coefficients else None,
        'Max Rank-Order Correlation Coefficient': max(rank_order_correlations) if rank_order_correlations else None,
        'Min Rank-Order Correlation Coefficient': min(rank_order_correlations) if rank_order_correlations else None,
    }
    
    return average_correlation, average_rank_order, description

    
            

def parse_result(data="correspondance_metrics.csv"):
    """parse the csv into latex table"""
    data = pd.read_csv(data)

    # Create the LaTeX code for the table
    latex_code = r"""
    \begin{table}[ht]
    \centering
    \caption{Image quality assessment result. CC: Correlation Coefficient; OR: Outlier Ratio; MAE: Mean Absolute Error; RMS: Root Mean Square Error; SORCC: Spearman's Rank Correlation Coefficient. We mark the \textbf{best results} with bold font and the \underline{second best} with underline.}
    \label{tab:IQA}
    \begin{tabular}{@{}lcccccccccc@{}}
    \toprule
    & \multicolumn{4}{c}{\textbf{Variance-Weighted Regression}} & \multicolumn{4}{c}{\textbf{Non-Linear Regression}} & \multicolumn{1}{c}{\textbf{Rank-Order}} \\ 
    \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-10}
    & CC $\uparrow$ & MAE $\downarrow$ & RMS $\downarrow$ & OR $\downarrow$ & CC $\uparrow$ & MAE $\downarrow$ & RMS $\downarrow$ & OR $\downarrow$ & SORCC $\uparrow$ \\ 
    \midrule
    """

    # Extract data for each row
    for index, row in data.iterrows():
        method = row.iloc[0]
        values = row.iloc[1:]
        if method in ["mae_normalized", "emd"]:
            continue


        
        # Highlight best and second-best results for each column
        formatted_values = []
        for col, value in enumerate(values):
            column_values = data.iloc[:, col + 1]
            best = column_values.max() if col % 4 == 0 else column_values.min()
            second_best = column_values.nlargest(2).iloc[-1] if col % 4 == 0 else column_values.nsmallest(2).iloc[-1]
            
            if value == best:
                formatted_values.append(f"\\textbf{{{value:.3f}}}")
            elif value == second_best:
                formatted_values.append(f"\\underline{{{value:.3f}}}")
            else:
                formatted_values.append(f"{value:.3f}")
        
        latex_code += f"{method} & " + " & ".join(formatted_values) + r" \\" + "\n"

    # Add bottom rules
    latex_code += r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
    
    print(latex_code)


if __name__ == "__main__":
    # take screenshots for all html files in the directories in results/
    # process_results(take_screenshot_for_dir)

    # initialize annotation tool
    # initialize_annotation_tool()
    
    # input: original_generated_pairs.csv output: annotation_tool/static/images/[dir_name]/[png_file]
    # results = pd.read_csv("original_generated_pairs.csv")
    # copy_pngs_to_annotation_tool(results.values.tolist())

    ############# Process annotations ############

    # input: annotation_tool/instance/annotation_tool.db output: annotation.csv
    # annotation_to_table().to_csv("./annotation.csv", index=False)

    # input: image_pairs.csv output: metrics.csv
    # evaluate_metrics()

    # input: annotation.csv output: normalized_annotation.csv
    # normalize_annotation()

    # input: normalized_annotation.csv, metrics.csv output: merged.csv
    # merge_result()

    # input: merged.csv output: correspondance_metrics.csv
    # get_correspondance_metrics().to_csv("correspondance_metrics.csv", index=True)

    # print(get_annotation_validity())

    # for norm_annot_bin in os.listdir():
    #     if norm_annot_bin.startswith("normalize_annotation_bin_"):
    #         merge_result(norm_annot_bin, "metrics.csv", output=f"merged_{norm_annot_bin}")
    #         get_correspondance_metrics(f"merged_{norm_annot_bin}").to_csv(f"correspondance_metrics_{norm_annot_bin}", index=True)
    
    
    parse_result()
    # for correspondance_metrics in os.listdir():
    #     if correspondance_metrics.startswith("correspondance_metrics_"):
    #         parse_result(correspondance_metrics)
    #         print("=====================================================")



    



    

    
    


    
        
