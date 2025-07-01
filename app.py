# app.py

import os
import torch
import pandas as pd
from PIL import Image
import clip
import gradio as gr
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test data
df = pd.read_csv("Dataset/test_split.csv")
categories = df['category'].unique().tolist()

# Load Models
original_pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
original_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(original_pipeline.scheduler.config)
original_pipeline.to(device)

finetuned_pipeline = DiffusionPipeline.from_pretrained(
    "Code/final_pipeline",  # make sure this path matches the location in your repo
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
finetuned_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(finetuned_pipeline.scheduler.config)
finetuned_pipeline.to(device)

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

def generate(category, image_index, model_choice):
    row = df[df['category'] == category].iloc[int(image_index)]
    input_image = Image.open(row["view_00"]).convert("RGB")
    gt_image = Image.open(row["multiview"]).convert("RGB")
    
    pipeline = original_pipeline if model_choice == "Original" else finetuned_pipeline
    output_image = pipeline(
        image=input_image,
        num_inference_steps=30,
        guidance_scale=3.0,
        strength=0.3,
        eta=0.2,
        output_type="pil"
    ).images[0]

    with torch.no_grad():
        input_feat = clip_model.encode_image(preprocess(gt_image).unsqueeze(0).to(device))
        output_feat = clip_model.encode_image(preprocess(output_image).unsqueeze(0).to(device))
        sim = torch.nn.functional.cosine_similarity(input_feat, output_feat).item()

    return input_image, output_image, gt_image, f"CLIP Similarity: {sim:.4f}"

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Dropdown(choices=categories, label="Category"),
        gr.Number(value=0, label="Image Index"),
        gr.Radio(["Original", "Fine-tuned"], label="Model")
    ],
    outputs=[
        gr.Image(label="Input View"),
        gr.Image(label="Generated View"),
        gr.Image(label="Ground Truth (Multiview)"),
        gr.Text(label="CLIP Similarity")
    ],
    title="Zero123++ Fine-Tuned vs Original Viewer",
    description="Compare outputs from fine-tuned and original Zero123++ models using ShapeNet samples"
)

demo.launch()
