# -*- coding: utf-8 -*-
# @Time : 5/15/23 11:17 PM
# @Author : AndresHG
# @File : nlpcloud_local.py
# @Email: andresherranz999@gmail.com

import pickle

import torch
from transformers import AutoTokenizer, GPTNeoXForQuestionAnswering

# Load vectorstore
with open("vectorstores_faiss/vectorstore_light.pkl", "rb") as f:
    vectorstore = pickle.load(f)
    print("Vectirstore loaded")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = GPTNeoXForQuestionAnswering.from_pretrained("EleutherAI/gpt-neox-20b")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

# target is "nice puppet"
target_start_index = torch.tensor([14])
target_end_index = torch.tensor([15])

outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss