import pandas as pd
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import os
from PIL import Image
import argparse
from openai import OpenAI
import requests
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Pick Scores")
    parser.add_argument(
        "--csv_file_path", 
        type=str, 
        default='./baseline_result/promptist/Parti_Expanded_Prompts.tsv', 
        help="Path to the CSV file containing input data"
    )
    parser.add_argument(
        "--expanded_image_dir", 
        type=str, 
        default='./baseline_save_result/Parti_Expanded_Prompts/', 
        help="Directory containing the expanded images"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    return parser.parse_args()
args = parse_args()

def set_seed(seed):
    """
    设置随机种子以确保结果一致
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(args.seed)
#set_seed(14)

client = OpenAI(
    api_key = "",
    base_url = "",
    default_headers = {"x-foo": "true"}
)

def complete_gpt3(prompt, model_name, n=1):
    response = None
    received = False
    while not received:
        try:
            response = client.images.generate(
                        model=model_name,
                        prompt=prompt,
                        n=n,
                        response_format='b64_json')
            received = True
        except Exception as error:
            if error.body['message'].startswith("Your request was rejected as a result of our safety system."):
                return -1
            print("An error occurred in complete_gpt3:", error) # An error occurred: name 'x' is not defined
            # time.sleep(3)
    return response


input_file = args.csv_file_path
expanded_save_dir = args.expanded_image_dir

os.makedirs(expanded_save_dir, exist_ok=True)

file_name, file_extension = os.path.splitext(input_file)


if file_extension=='.csv':
    df = pd.read_csv(input_file, sep=',')
    optimized_prompts = df['Optimized_Prompt'].tolist()
elif file_extension=='.tsv':
    df = pd.read_csv(input_file, sep='\t')
    optimized_prompts = df['expanded_prompt'].tolist()

def generate_images(prompt, save_dir, filename_prefix, num_images=3):
    with autocast("cuda"):
        g_images = []
        for i in range(num_images):
            response = complete_gpt3(prompt, 'dall-e-3', 1)
            if response==-1:
                print(f'{prompt} is not safe!')
                return 0
            g_images.append(response.data[0].b64_json)
        for i in range(num_images):
            image_data = base64.b64decode(g_images[i])
            img_data = Image.open(BytesIO(image_data))
            # img_data = requests.get(g_images[i]).content
            image_path = os.path.join(save_dir, f"{filename_prefix}_{i}.png")
            img_data.save(image_path, "PNG")

for idx, (expanded_prompt) in tqdm(enumerate(optimized_prompts), total=len(optimized_prompts), leave=False):
    generate_images(expanded_prompt, expanded_save_dir, f"expanded_{idx}")
print(f"Images saved to:\n- Expanded: {expanded_save_dir}")