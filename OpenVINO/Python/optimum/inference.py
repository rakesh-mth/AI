from optimum.intel import OVModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline

model = OVModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad-ov-fp32")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
ov_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device_map="auto")
result = ov_pipe("What is OpenVINO?", "OpenVINO is a framework for deep learning inference optimization")

print(result)