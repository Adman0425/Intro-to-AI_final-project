import json
from tokenizers import ByteLevelBPETokenizer

# 加载 JSONL 文件并提取文本数据
texts = []
with open("train_no_robot.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        for message in data["messages"]:
            texts.append(message["content"])

# 初始化并训练 ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(texts, vocab_size=30000, min_frequency=2)

# 保存 tokenizer
tokenizer.save_model("tokenized_data\example-train")
