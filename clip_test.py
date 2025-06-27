import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np

# 配置
MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_DIR = "data/images"

# 1. 加载模型
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print(f"Loaded {MODEL_NAME} model")

# 2. 加载女友的图片库
image_paths = []
images = []
for img_file in os.listdir(IMAGE_DIR):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(IMAGE_DIR, img_file)
        try:
            image = Image.open(img_path)
            image_paths.append(img_path)
            images.append(image)
            print(f"Loaded: {img_file}")
        except:
            print(f"Failed to load: {img_file}")

if not images:
    print("No valid images found! Please add images to data/images")
    exit()

# 3. 测试1：文本搜索
def test_text_search(query):
    print(f"\nTesting text search: '{query}'")
    
    # 处理输入
    inputs = processor(
        text=[query], 
        images=images, 
        return_tensors="pt", 
        padding=True
    )
    
    # 模型推理
    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text
    
    # 计算概率
    probs = logits_per_text.softmax(dim=-1).detach().numpy()[0]
    
    # 打印结果
    print("Top results:")
    sorted_indices = np.argsort(probs)[::-1]  # 从高到低排序
    for idx in sorted_indices[:3]:  # 显示前三结果
        print(f"  {image_paths[idx]} - Probability: {probs[idx]:.4f}")
    
    # 显示图片
    best_idx = sorted_indices[0]
    best_image = images[best_idx]
    best_image.show(title=f"Top match for '{query}'")

# 4. 测试2：图搜图
def test_image_search(query_image_path):
    print(f"\nTesting image search with: {query_image_path}")
    
    # 加载查询图片
    try:
        query_image = Image.open(query_image_path)
        query_image.show(title="Query Image")
    except:
        print("Failed to load query image")
        return
    
    # 提取查询图片特征
    inputs = processor(images=query_image, return_tensors="pt")
    query_features = model.get_image_features(**inputs)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    query_features = query_features.detach().numpy()
    
    # 提取图库图片特征
    library_features = []
    for image in images:
        inputs = processor(images=image, return_tensors="pt")
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        library_features.append(features.detach().numpy())
    
    library_features = np.vstack(library_features)
    
    # 计算相似度
    similarity = np.dot(query_features, library_features.T)[0]
    
    # 打印结果
    print("Top similar images:")
    sorted_indices = np.argsort(similarity)[::-1]
    for idx in sorted_indices[:3]:
        print(f"  {image_paths[idx]} - Similarity: {similarity[idx]:.4f}")
        images[idx].show(title=f"Match #{sorted_indices.tolist().index(idx)+1}")

# 5. 运行测试
if __name__ == "__main__":
    # 测试文本搜索
    test_text_search("enclosed human body")
    test_text_search("emotional coldness")
    test_text_search("ritualized dining behavior")
    test_text_search("obsessive routine")
    test_text_search("cold architectural order")
 



    # 测试图搜图（使用第一张图片作为查询）
    if image_paths:
        test_image_search(image_paths[0])
    
    print("\nPhase 1 testing complete! Ready for CLI tool development.")