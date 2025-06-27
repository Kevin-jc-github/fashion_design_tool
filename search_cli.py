import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse
import time
import json
from pathlib import Path

# 配置参数
MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_DIR = "data/images/"  # 图片库目录
FEATURES_FILE = "data/image_features.json"  # 特征缓存文件

# 确保目录存在
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

# 加载模型和处理器
print(f"⏳ Loading CLIP model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print(f"✅ Model loaded on {device} device")

def precompute_image_features(image_dir, features_file):
    """预计算图片库的特征向量并存储"""
    image_paths = []
    image_features = []
    
    print(f"🔍 Scanning image directory: {image_dir}")
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    # 如果特征文件存在且图片库未更新，直接加载
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            cache = json.load(f)
            cache_mtime = cache.get("last_modified", 0)
            current_mtime = os.path.getmtime(image_dir)
            
            # 检查目录是否修改过
            if cache_mtime >= current_mtime and set(cache["image_paths"]) == set(
                [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
            ):
                print("♻️ Loading precomputed features from cache")
                return cache["image_paths"], np.array(cache["features"])
    
    # 需要重新计算特征
    print("🔄 Computing features for images...")
    start_time = time.time()
    
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(valid_extensions):
            img_path = os.path.join(image_dir, img_file)
            try:
                image = Image.open(img_path)
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)  # 归一化
                image_paths.append(img_path)
                image_features.append(features.cpu().numpy().tolist()[0])  # 转为列表存储
                print(f"  ✔ Processed: {img_file}")
            except Exception as e:
                print(f"  ❌ Error processing {img_file}: {e}")
    
    # 转换为numpy数组
    image_features_np = np.array(image_features)
    
    # 保存到缓存文件
    cache_data = {
        "last_modified": os.path.getmtime(image_dir),
        "image_paths": [os.path.basename(p) for p in image_paths],
        "features": image_features_np.tolist()
    }
    
    with open(features_file, 'w') as f:
        json.dump(cache_data, f)
    
    print(f"💾 Saved features to {features_file}")
    print(f"⏱️ Feature computation took {time.time()-start_time:.2f} seconds")
    
    return image_paths, image_features_np

def text_search(query_text, all_image_paths, all_image_features, top_k=5):
    """文本搜索函数"""
    print(f"\n🔎 Searching for: '{query_text}'")
    start_time = time.time()
    
    # 处理文本
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化
    text_features = text_features.cpu().numpy()

    # 计算相似度 (余弦相似度)
    similarities = np.dot(text_features, all_image_features.T)[0]
    
    # 获取相似度最高的top_k个索引
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # 打印结果
    print(f"🏆 Top {top_k} results:")
    results = []
    for rank, idx in enumerate(top_indices, 1):
        img_path = all_image_paths[idx]
        score = similarities[idx]
        print(f"  #{rank}: {img_path} - Similarity: {score:.4f}")
        results.append({"path": img_path, "score": score})
    
    print(f"⏱️ Search took {time.time()-start_time:.2f} seconds")
    return results

def image_search(query_image_path, all_image_paths, all_image_features, top_k=5):
    """图搜图函数"""
    print(f"\n🖼️ Searching with image: {query_image_path}")
    start_time = time.time()
    
    try:
        # 加载并处理查询图片
        query_image = Image.open(query_image_path)
        inputs = processor(images=query_image, return_tensors="pt").to(device)
        with torch.no_grad():
            query_features = model.get_image_features(**inputs)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
        query_features = query_features.cpu().numpy()

        # 计算相似度
        similarities = np.dot(query_features, all_image_features.T)[0]
        
        # 获取相似度最高的top_k个索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 打印结果
        print(f"🏆 Top {top_k} results:")
        results = []
        for rank, idx in enumerate(top_indices, 1):
            img_path = all_image_paths[idx]
            score = similarities[idx]
            print(f"  #{rank}: {img_path} - Similarity: {score:.4f}")
            results.append({"path": img_path, "score": score})
        
        print(f"⏱️ Search took {time.time()-start_time:.2f} seconds")
        return results
    except Exception as e:
        print(f"❌ Error processing query image: {e}")
        return []

def print_help():
    """打印帮助信息"""
    print("\n🎨 Fashion Visionary CLI Tool")
    print("Usage:")
    print("  Search by text: python search_cli.py --text \"design keywords\" [--topk 5]")
    print("  Search by image: python search_cli.py --image path/to/image.jpg [--topk 5]")
    print("  Update image library: python search_cli.py --update")
    print("\nExamples:")
    print("  python search_cli.py --text \"deconstructed futurism\"")
    print("  python search_cli.py --image data/images/texture_sample.jpg --topk 3")
    print("  python search_cli.py --update")

def main():
    """主函数：解析命令行参数"""
    parser = argparse.ArgumentParser(description="Fashion Visionary CLI Search Tool", add_help=False)
    parser.add_argument("--text", type=str, help="Text query for image search")
    parser.add_argument("--image", type=str, help="Path to query image for search")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to show (default: 5)")
    parser.add_argument("--update", action="store_true", help="Force update image features cache")
    parser.add_argument("--help", action="store_true", help="Show help message")
    
    args = parser.parse_args()
    
    if args.help:
        print_help()
        return
    
    # 强制更新缓存
    if args.update:
        if os.path.exists(FEATURES_FILE):
            os.remove(FEATURES_FILE)
            print("♻️ Feature cache cleared")
    
    # 预计算/加载特征
    print("\n" + "="*50)
    print("📂 Preparing image library...")
    all_image_paths, all_image_features = precompute_image_features(IMAGE_DIR, FEATURES_FILE)
    
    if not all_image_paths:
        print("❌ No valid images found in library!")
        print("Please add images to data/images directory")
        return
    
    print(f"📊 Library contains {len(all_image_paths)} images")
    print("="*50 + "\n")
    
    # 执行搜索
    if args.text:
        text_search(args.text, all_image_paths, all_image_features, args.topk)
    elif args.image:
        image_search(args.image, all_image_paths, all_image_features, args.topk)
    else:
        print("⚠️ Please provide a search query (--text or --image)")
        print("Use --help for usage instructions")

if __name__ == "__main__":
    main()