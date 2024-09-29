import torch
from optimum.intel import OVModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline

model = OVModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad-ov-fp32")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad-ov-fp32")

question, text = "What is OpenVINO?", "OpenVINO is a framework for deep learning inference optimization"

inputs = tokenizer(question, text, return_tensors="pt")
outputs = model(**inputs)

answer_start_index = torch.argmax(outputs.start_logits, axis=-1).item()
answer_end_index = torch.argmax(outputs.end_logits, axis=-1).item()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
result = tokenizer.decode(predict_answer_tokens)

print(result)

