import requests
import os
import time

# 设置关键词与对应英文翻译
keywords = {
    "limitation": "局限",
    "coldness": "冰冷",
    "ritualized dining behavior": "餐桌礼仪",
    "obsessive routine": "强迫性重复"
}

# API 端点
SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"

# 下载保存路径
SAVE_ROOT = "met_images"
os.makedirs(SAVE_ROOT, exist_ok=True)

for eng_keyword, cn_keyword in keywords.items():
    print(f"\n🔍 Processing keyword: {eng_keyword} ({cn_keyword})")

    # 创建保存目录
    keyword_dir = os.path.join(SAVE_ROOT, eng_keyword.replace(" ", "_"))
    os.makedirs(keyword_dir, exist_ok=True)

    # 发起搜索请求
    params = {"q": eng_keyword, "hasImages": "true"}
    try:
        response = requests.get(SEARCH_URL, params=params, timeout=10)
        if response.status_code == 200 and response.text.strip():
            search_resp = response.json()
            object_ids = search_resp.get("objectIDs", [])[:30]  # 限制最多30个
            print(f"🔢 Found {len(object_ids)} objects for '{eng_keyword}'")
        else:
            print(f"❌ Search failed with status {response.status_code}")
            continue
    except Exception as e:
        print(f"❌ Error searching for '{eng_keyword}': {e}")
        continue

    # 下载每个 object 的图片
    for obj_id in object_ids:
        try:
            object_response = requests.get(f"{OBJECT_URL}{obj_id}", timeout=10)
            if object_response.status_code != 200 or not object_response.text.strip():
                print(f"⚠️ Failed to get object {obj_id}: status {object_response.status_code}")
                continue

            obj_resp = object_response.json()
            image_url = obj_resp.get("primaryImage")

            if image_url:
                img_data = requests.get(image_url, timeout=10).content
                img_name = os.path.join(keyword_dir, f"{obj_id}.jpg")
                with open(img_name, 'wb') as f:
                    f.write(img_data)
                print(f"✅ Saved: {img_name}")
            else:
                print(f"⛔ No image for object {obj_id}")

            time.sleep(0.5)  # 防止过快被限流

        except Exception as e:
            print(f"⚠️ Error on object ID {obj_id}: {e}")
            continue

print("\n✅ All keywords processed!")
