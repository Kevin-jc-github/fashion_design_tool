import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse
import time
import pickle
import json
from pathlib import Path

# 配置参数
MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_DIR = "data/images/"
FEATURES_FILE = "data/image_features.pkl"

# 确保目录存在
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

# 加载模型和处理器
print(f"⏳ Loading CLIP model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print(f"✅ Model loaded on {device} device")


def precompute_image_features(image_dir, features_file):
    """预计算图像特征并缓存"""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    if os.path.exists(features_file):
        print("♻️ Loading cached features from pickle")
        with open(features_file, "rb") as f:
            return pickle.load(f)

    print("🔄 Computing image features...")
    features = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                img_feat = model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            features.append(img_feat.cpu().numpy()[0])
            print(f"  ✔ Processed: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"  ❌ Error processing {img_path}: {e}")

    features_np = np.vstack(features)
    with open(features_file, "wb") as f:
        pickle.dump((image_paths, features_np), f)

    print(f"💾 Saved feature cache to {features_file}")
    return image_paths, features_np


def text_to_features(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)


def image_to_features(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)


def hybrid_search(text, image_path, all_paths, all_features, top_k=5):
    print(f"\n🔍 Hybrid search: \"{text}\" + image \"{image_path}\"")
    text_feat = text_to_features(text)
    image_feat = image_to_features(image_path)
    combined_feat = (text_feat + image_feat) / 2
    similarities = np.dot(combined_feat.cpu().numpy(), all_features.T)[0]
    return show_top_k(similarities, all_paths, top_k)


def text_search(text, all_paths, all_features, top_k=5):
    print(f"\n🔎 Searching by text: \"{text}\"")
    features = text_to_features(text)
    similarities = np.dot(features.cpu().numpy(), all_features.T)[0]
    return show_top_k(similarities, all_paths, top_k)


def image_search(image_path, all_paths, all_features, top_k=5):
    print(f"\n🖼️ Searching by image: {image_path}")
    features = image_to_features(image_path)
    similarities = np.dot(features.cpu().numpy(), all_features.T)[0]
    return show_top_k(similarities, all_paths, top_k)


def show_top_k(similarities, paths, top_k):
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    print(f"🏆 Top {top_k} results:")
    for rank, idx in enumerate(top_indices, 1):
        path = paths[idx]
        score = similarities[idx]
        print(f"  #{rank}: {path} - Similarity: {score:.4f}")
        results.append({"rank": rank, "path": path, "similarity": round(float(score), 4)})
    return results


def print_help():
    print("\n🎨 Fashion Visionary CLI Tool")
    print("Usage:")
    print("  --text \"your prompt\"")
    print("  --image path/to/image.jpg")
    print("  --topk 5")
    print("  --output results.json")
    print("Examples:")
    print("  python search_cli.py --text \"ritual and space\"")
    print("  python search_cli.py --image data/images/pic1.jpg")
    print("  python search_cli.py --text \"cold order\" --image data/images/pic1.jpg")
    print("  python search_cli.py --update")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text query")
    parser.add_argument("--image", type=str, help="Query image path")
    parser.add_argument("--topk", type=int, default=5, help="Top K results to show")
    parser.add_argument("--output", type=str, help="Save results to output JSON")
    parser.add_argument("--update", action="store_true", help="Force clear feature cache")
    parser.add_argument("--help", action="store_true", help="Show help")

    args = parser.parse_args()

    if args.help:
        print_help()
        return

    if args.update and os.path.exists(FEATURES_FILE):
        os.remove(FEATURES_FILE)
        print("♻️ Cache cleared.")

    print("\n📂 Preparing image library...")
    all_paths, all_features = precompute_image_features(IMAGE_DIR, FEATURES_FILE)
    print(f"📊 Loaded {len(all_paths)} images\n")

    if not args.text and not args.image:
        print("⚠️ Please provide --text and/or --image as query input.")
        return

    if args.text and args.image:
        results = hybrid_search(args.text, args.image, all_paths, all_features, args.topk)
    elif args.text:
        results = text_search(args.text, all_paths, all_features, args.topk)
    elif args.image:
        results = image_search(args.image, all_paths, all_features, args.topk)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📝 Results saved to {args.output}")


if __name__ == "__main__":
    main()
