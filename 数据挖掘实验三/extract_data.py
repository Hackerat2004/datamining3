import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def process_xml_file(gz_file_path):
    """
    处理单个gz文件，提取文章标题和摘要
    """
    articles = []
    
    # 使用gzip打开压缩文件
    with gzip.open(gz_file_path, 'rb') as gz_file:
        # 解析XML
        tree = ET.parse(gz_file)
        root = tree.getroot()
        
        # 遍历所有文章
        for article in root.findall('.//PubmedArticle'):
            try:
                # 查找标题
                title_element = article.find('.//ArticleTitle')
                title = title_element.text if title_element is not None else ""
                
                # 查找摘要
                abstract_element = article.find('.//Abstract/AbstractText')
                abstract = abstract_element.text if abstract_element is not None else ""
                
                # 如果标题或摘要不为空，则添加到结果中
                if title or abstract:
                    articles.append({
                        'title': title,
                        'abstract': abstract
                    })
            except Exception as e:
                print(f"处理文章时出错: {str(e)}")
                continue
    
    return articles

def main():
    # 指定gz文件所在目录
    input_dir = Path('pubmed_files')
    # 指定输出JSON文件路径
    output_file = 'pubmed_articles.json'
    
    all_articles = []
    
    # 获取所有gz文件
    gz_files = list(input_dir.glob('*.xml.gz'))
    
    # 使用tqdm显示处理进度
    for gz_file in tqdm(gz_files, desc="处理文件"):
        print(f"\n处理文件: {gz_file.name}")
        articles = process_xml_file(gz_file)
        all_articles.extend(articles)
        print(f"从 {gz_file.name} 中提取了 {len(articles)} 篇文章")
    
    # 保存为JSON文件
    print(f"\n正在保存到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理 {len(gz_files)} 个文件，提取了 {len(all_articles)} 篇文章")

if __name__ == "__main__":
    main()