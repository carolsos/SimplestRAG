# SimplestRAG

只涉及模型的Inference，不涉及预训练

以下模型和数据，默认在CPU上运行，如果想转移到GPU，得使用torch（用KIMI or 豆包查一下）

## 使用Langchain、embedding、chroma版（比较新的东西）

### 导入相关库
使用Chroma作为向量数据库
```python
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
```

### 路径配置
使用绝对路径
```python
embedding_model_path = "/xxx/e5-base-v2"
corpus_path = "/xxx/corpus/general_knowledge.jsonl"
```

### 读取语料库
```python
def read_jsonl(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            class DocWrapper:
                def __init__(self, content):
                    self.page_content = content
                    self.metadata = {}
            docs.append(DocWrapper(doc['contents']))
    return docs

docs = read_jsonl(corpus_path)
```

### 文本分割
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
documents = text_splitter.split_documents(docs)  # docs 已经是字符串列表
```

### 嵌入模型
```python
embedding_model = HuggingFaceBgeEmbeddings(model_name=embedding_model_path)
```

### 构建向量数据库
```python
vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory="./vectorstore")
```

### 查询
```python
query = "Who was the first winner of the Nobel Prize?"
result = vectorstore.similarity_search(query, k=3)
```

### 打印检索出来的 context
```python
for doc in result:
    print(doc.page_content)
    print("********")
```

### 使用本地模型
使用绝对路径
```python
local_model_path = "/xxx/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 设置填充 token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(local_model_path)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

context = format_docs(result)

def generate_response(context, query):
    prompt = context + "\n\nQuestion: " + query
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# RAG 输出结果
rag_response = generate_response(context, query)
print("RAG 输出结果:", rag_response)

# 直接使用 LLM 输出结果
llm_prompt = "Question: " + query
llm_inputs = tokenizer(llm_prompt, return_tensors='pt', padding=True, truncation=True)
llm_input_ids = llm_inputs['input_ids']
llm_attention_mask = llm_inputs['attention_mask']
llm_outputs = model.generate(input_ids=llm_input_ids, attention_mask=llm_attention_mask)
llm_response = tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
print("LLM 输出结果：", llm_response)
```

### 完整版代码
```python
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer

# 路径配置
embedding_model_path = "/xxx/e5-base-v2"
corpus_path = "/xxx/corpus/general_knowledge.jsonl"

# 读取语料库
def read_jsonl(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            class DocWrapper:
                def __init__(self, content):
                    self.page_content = content
                    self.metadata = {}
            docs.append(DocWrapper(doc['contents']))
    return docs

docs = read_jsonl(corpus_path)

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
documents = text_splitter.split_documents(docs)  # docs 已经是字符串列表

# 嵌入模型
embedding_model = HuggingFaceBgeEmbeddings(model_name=embedding_model_path)

# 构建向量数据库
vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory="./vectorstore")

# 查询
query = "Who was the first winner of the Nobel Prize?"
result = vectorstore.similarity_search(query, k=3)

# 打印检索出来的 context
for doc in result:
    print(doc.page_content)
    print("********")

# 使用本地模型
local_model_path = "/xxx/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 设置填充 token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(local_model_path)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

context = format_docs(result)

def generate_response(context, query):
    prompt = context + "\n\nQuestion: " + query
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# RAG 输出结果
rag_response = generate_response(context, query)
print("RAG 输出结果:", rag_response)

# 直接使用 LLM 输出结果
llm_prompt = "Question: " + query
llm_inputs = tokenizer(llm_prompt, return_tensors='pt', padding=True, truncation=True)
llm_input_ids = llm_inputs['input_ids']
llm_attention_mask = llm_inputs['attention_mask']
llm_outputs = model.generate(input_ids=llm_input_ids, attention_mask=llm_attention_mask)
llm_response = tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
print("LLM 输出结果：", llm_response)
```

## 传统nlp版，使用TF-IDF和cosine_similarity

```python
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 路径配置
corpus_path = "/xxx/corpus/general_knowledge.jsonl"
local_model_path = "/xxx/Llama-2-7b-chat-hf"

# 读取语料库
def read_jsonl(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            docs.append(doc['contents'])
    return docs

docs = read_jsonl(corpus_path)

# 文本向量化
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs)

# 查询
query = "Who was the first winner of the Nobel Prize?"
query_vector = vectorizer.transform([query])

# 相似度计算
similarities = cosine_similarity(query_vector, doc_vectors).flatten()
top_indices = similarities.argsort()[-3:][::-1]
top_docs = [docs[i] for i in top_indices]

# 打印检索出来的 context
for doc in top_docs:
    print(doc)
    print("********")

# 使用本地模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 设置填充 token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(local_model_path)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# RAG 输出结果
rag_prompt = "\n\n".join(top_docs) + "\n\nQuestion: " + query
rag_response = generate_response(rag_prompt)
print("RAG 输出结果:", rag_response)

# 直接使用 LLM 输出结果
llm_prompt = "Question: " + query
llm_response = generate_response(llm_prompt)
print("LLM 输出结果：", llm_response)
```
