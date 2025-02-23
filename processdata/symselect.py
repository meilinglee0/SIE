import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import re
import json
import pandas as pd

# 2. 对文本进行分词和编码
def encode_texts(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取 [CLS] token 的表示作为文本的特征
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token 的表示
    return cls_embeddings.cpu().numpy()



def find_symptom_indices(text, symptoms):
    """
    检查文本是否包含在 symptoms 列表中的任何一个词，返回这些词的索引列表。
    
    Args:
    - text (str): 要检查的文本。
    - symptoms (list): 包含症状词的列表。
    
    Returns:
    - List[int]: 包含找到的症状词在列表中的索引。
    """
    # 结果列表，存储找到的症状词的索引
    indices = []
    
    # 遍历所有症状词
    for i, symptom in enumerate(symptoms):
        # 如果症状词在文本中出现，则将其索引添加到结果列表
        if symptom.replace('_', ' ').lower() in text.lower():
            indices.append(i)
    
    return indices

# 定义症状列表
symptoms = [
    "Anger_Irritability", "Anxious_Mood", "Autonomic_symptoms", "Cardiovascular_symptoms",
    "Catatonic_behavior", "Decreased_energy_tiredness_fatigue", "Depressed_Mood",
    "Gastrointestinal_symptoms", "Genitourinary_symptoms", "Hyperactivity_agitation",
    "Impulsivity", "Inattention", "Indecisiveness", "Respiratory_symptoms", "Suicidal_ideas",
    "Worthlessness_and_guilty", "avoidance_of_stimuli", "compensatory_behaviors_to_prevent_weight_gain",
    "compulsions", "diminished_emotional_expression", "do_things_easily_get_painful_consequences",
    "drastical_shift_in_mood_and_energy", "fear_about_social_situations", "fear_of_gaining_weight",
    "fears_of_being_negatively_evaluated", "flight_of_ideas", "intrusion_symptoms",
    "loss_of_interest_or_motivation", "more_talktive", "obsession", "panic_fear", "pessimism",
    "poor_memory", "sleep_disturbance", "somatic_muscle", "somatic_symptoms_others",
    "somatic_symptoms_sensory", "weight_and_appetite_change",'none'
]

# 替换下划线为白色空格
formatted_symptoms = [symptom.replace('_', ' ') for symptom in symptoms]
diseases = [ "adhd", "anxiety", "bipolar disorder", "depression", "eating disorder", "ocd", "ptsd", "health"]
# 从本地路径加载模型和tokenizer
model_path = '....'
tokenizer_path = '....'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path)
# 将模型移动到 GPU（如果可用）或保持在 CPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
syms = []
ds= []
for formatted_symptom in formatted_symptoms:
    temp_s = encode_texts(formatted_symptom)
    syms.append(temp_s)
for d in diseases:
    temp_d = encode_texts(d)
    ds.append(temp_d)

symst = np.array(syms).reshape(-1,768)
np.save('.....', symst)
