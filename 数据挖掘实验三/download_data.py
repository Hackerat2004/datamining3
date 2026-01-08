import requests
import os
from tqdm import tqdm

def download_file(url, output_path):
    """
    下载文件并显示进度条
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # 创建保存文件的目录
    output_dir = 'pubmed_files'
    os.makedirs(output_dir, exist_ok=True)
    
    # 基础URL
    base_url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    
    # 下载前10个文件
    for i in range(1, 11):
        # PubMed文件命名格式：pubmed25n0001.xml.gz
        filename = f"pubmed25n{i:04d}.xml.gz"
        url = base_url + filename
        output_path = os.path.join(output_dir, filename)
        
        print(f"正在下载文件 {i}/10: {filename}")
        try:
            download_file(url, output_path)
            print(f"成功下载: {filename}")
        except Exception as e:
            print(f"下载 {filename} 时发生错误: {str(e)}")

if __name__ == "__main__":
    main()
