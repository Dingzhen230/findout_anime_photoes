import os
import requests
import time
from urllib.parse import quote
import mimetypes
from pathlib import Path
import shutil
from random import randint

def make_dataset(path):
    folder_path = Path(path)
    if folder_path.exists():
        for item in folder_path.iterdir():
            if item.is_file():
                item.unlink()  # 删除文件
            elif item.is_dir():
                shutil.rmtree(item)  # 删除子目录
        print(f"已清空文件夹: {path}")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "train", "gt"), exist_ok=True)
    os.makedirs(os.path.join(path, "train", "image"), exist_ok=True)
    os.makedirs(os.path.join(path, "test"), exist_ok=True)
    os.makedirs(os.path.join(path, "test", "gt"), exist_ok=True)
    os.makedirs(os.path.join(path, "test", "image"), exist_ok=True)
    os.makedirs(os.path.join(path, "val"), exist_ok=True)
    os.makedirs(os.path.join(path, "val", "gt"), exist_ok=True)
    os.makedirs(os.path.join(path, "val", "image"), exist_ok=True)

def download_baidu_images(keyword, label, n, path):
    """
    从百度图片搜索下载指定数量的图片
    :param keyword: 搜索关键词
    :param n: 需要下载的图片数量
    :param path: 图片保存路径
    """
    # 请求头设置
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://image.baidu.com/'
    }
    
    # 初始化参数
    downloaded = 0
    pn = 0  # 当前页码偏移量
    retry_limit = 3
    print(f"开始下载关键词：{keyword}，目标数量：{n}，保存路径：{path}")
    global total
    
    while downloaded < n:
        # 计算本次请求需要获取的数量
        rn = min(30, n - downloaded)
        
        # 构造请求参数
        params = {
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'fr': '',
            'word': keyword,
            'queryWord': keyword,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'st': -1,
            'istype': 2,  # 2表示原图
            'pn': pn,
            'rn': rn,
            'gsm': '1e',
            'timestamp': int(time.time() * 1000)
        }
        
        try:
            # 发送搜索请求
            search_url = 'https://image.baidu.com/search/acjson'
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            # 解析JSON数据
            data = response.json().get('data', [])
            
            if not data:
                continue
            print(f"获取到 {len(data)} 条数据")
            
            for item in data:
                if downloaded >= n:
                    break
                if not item:
                    continue
                # 获取图片URL
                img_url = item.get('middleURL')
                # 处理特殊字符转义
                img_url = requests.utils.unquote(img_url)
                
                # 下载图片
                for _ in range(retry_limit):
                    try:
                        img_res = requests.get(img_url, headers={'Referer': 'https://image.baidu.com/'}, timeout=10)
                        img_res.raise_for_status()
                        
                        # 获取文件扩展名
                        content_type = img_res.headers.get('Content-Type', '')
                        ext = mimetypes.guess_extension(content_type.split(';')[0].strip()) or '.jpg'
                        ext = ext if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'] else '.jpg'
                        
                        # 生成文件名
                        filename = f"{total+1:04d}{ext}"
                        rand = randint(0, 10000)
                        if rand >= 3000:
                            save_path = os.path.join(path, "train", "image", filename)
                        elif rand >= 1000:
                            save_path = os.path.join(path, "val", "image", filename)
                        else:
                            save_path = os.path.join(path, "test", "image", filename)
                        label_path = save_path.replace("image", "gt").replace("jpg", "txt")
                        
                        # 保存文件
                        with open(save_path, 'wb') as f:
                            f.write(img_res.content)
                        with open(label_path, 'w') as f:
                            f.write(str(label))
                        
                        total += 1
                        downloaded += 1
                        print(f"成功下载：{filename}（{downloaded}/{n}）")
                        break
                    except Exception as e:
                        print(f"下载失败 {img_url}，重试 {_+1}/{retry_limit}。错误：{str(e)}")
                        time.sleep(1)
                
                # time.sleep(0.5)  # 防止请求过快
                
            pn += rn
            
        except Exception as e:
            print(f"搜索请求失败：{str(e)}")
            break

if __name__ == "__main__":
    dataset_path = "./dataset"  # 保存路径
    total = 0
    make_dataset(dataset_path)
    download_baidu_images("动漫图片", 1, 30, dataset_path)
    download_baidu_images("聊天记录截图", 2, 30, dataset_path)
    download_baidu_images("人像照片", 3, 30, dataset_path)
    download_baidu_images("风景照片", 4, 30, dataset_path)