import json
import re
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_hf_mirrors():
    """设置Hugging Face镜像，加速模型下载"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'
    
# 设置镜像
set_hf_mirrors()

class MedicalEntityExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B", device=None):
        """
        初始化医学实体提取器
        
        参数:
            model_name: Qwen模型名称
            device: 计算设备
        """
        self.model_name = model_name
        
        # 检查GPU是否真正可用
        gpu_available = torch.cuda.is_available()
        try:
            if gpu_available:
                # 尝试获取GPU信息，如果失败则认为GPU不可用
                torch.cuda.get_device_name(0)
        except:
            gpu_available = False
            
        self.device = device if device else torch.device('cuda' if gpu_available else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 根据设备决定是否使用half precision
        if str(self.device) == 'cpu':
            print("使用CPU运行模型，将使用全精度")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            print("使用GPU运行模型，将使用半精度")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16  # 使用半精度以节省内存
            )
        
        # 设置token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("模型加载完成")
    
    def extract_entities(self, text):
        """
        从文本中提取疾病实体
        
        参数:
            text: 输入文本
                
        返回:
            dict: 包含提取的疾病实体
        """
        # 简化的提示模板 - 仅提取疾病
        prompt = f"""Extract disease names from the following medical text.
Return them in the following JSON format:
{{
  "diseases": []
}}

If you find any diseases or medical conditions, list them inside the array. If no diseases are found, return an empty array.

Medical text:
{text}

JSON output:
"""
        
        # 生成结果
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=512,  # 减少生成长度
                temperature=0.1,
                top_p=0.7,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 改进的JSON提取和错误处理
        try:
            # 尝试找到JSON部分
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end <= json_start:
                print("无法在响应中找到有效的JSON结构")
                print(f"响应内容: {response}")
                return {"diseases": []}
            
            # 提取JSON字符串
            json_str = response[json_start:json_end]
            
            # 尝试清理JSON字符串
            json_str = self.clean_json_string(json_str)
            
            # 解析JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"问题的JSON字符串: {json_str}")
                # 如果出错，尝试一个更直接的方法提取疾病列表
                diseases = self.extract_diseases_manually(response)
                return {"diseases": diseases}
        except Exception as e:
            print(f"处理JSON时出错: {str(e)}")
            print(f"响应内容: {response}")
            return {"diseases": []}

    def clean_json_string(self, json_str):
        """清理JSON字符串，修复常见问题"""
        # 替换单引号为双引号
        if '"' not in json_str and "'" in json_str:
            json_str = json_str.replace("'", '"')
        
        # 处理尾部逗号问题
        json_str = json_str.replace(",]", "]").replace(",}", "}")
        
        # 处理省略号问题
        json_str = json_str.replace("...", "")
        
        # 确保键名使用双引号
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
        
        return json_str

    def extract_diseases_manually(self, text):
        """手动从响应中提取疾病列表，适用于JSON解析失败的情况"""
        diseases = []
        try:
            # 尝试使用正则表达式找到方括号内的内容
            brackets_pattern = r'"diseases"\s*:\s*\[(.*?)\]'
            match = re.search(brackets_pattern, text, re.DOTALL)
            
            if match:
                content = match.group(1).strip()
                if content:
                    # 分割并清理引号内的内容
                    items = re.findall(r'"([^"]*)"', content)
                    diseases = [item.strip() for item in items if item.strip()]
        except Exception as e:
            print(f"手动提取疾病失败: {str(e)}")
        
        return diseases

def process_json_data(input_file, extractor, max_articles=None):
    """
    处理预先准备好的JSON文件，提取实体
    
    参数:
        input_file: 输入JSON文件路径
        extractor: 实体提取器
        max_articles: 最多处理的文章数（None表示处理所有）
        
    返回:
        list: 包含提取实体的文章列表
    """
    print(f"加载JSON数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    
    # 限制处理数量
    if max_articles is not None:
        articles_data = articles_data[:max_articles]
    
    # 提取实体
    articles_with_entities = []
    
    for article in tqdm(articles_data, desc="处理文章"):
        try:
            title = article.get('title', '')
            abstract = article.get('abstract', '')
            
            # 如果标题或摘要不为空
            if title or abstract:
                # 组合文本
                text = f"标题: {title}\n摘要: {abstract}"
                
                # 提取实体
                entities = extractor.extract_entities(text)
                
                # 添加到结果中
                article_with_entities = article.copy()  # 复制原文章数据
                article_with_entities['diseases'] = entities.get('diseases', [])  # 只保存疾病
                articles_with_entities.append(article_with_entities)
                
        except Exception as e:
            print(f"处理文章时出错: {str(e)}")
            continue
    
    return articles_with_entities

def main():
    # 指定输入JSON文件路径
    input_file = 'pubmed_articles.json'
    # 指定输出JSON文件路径
    output_file = 'pubmed_entities.json'
    
    # 最多处理的文章数（可以调整或设置为None处理所有文章）
    max_articles = 100
    
    # 初始化实体提取器
    extractor = MedicalEntityExtractor()
    
    # 处理JSON数据并提取实体
    articles_with_entities = process_json_data(input_file, extractor, max_articles)
    
    # 保存为JSON文件
    print(f"\n正在保存到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles_with_entities, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共提取了 {len(articles_with_entities)} 篇文章的实体")

if __name__ == "__main__":
    main()