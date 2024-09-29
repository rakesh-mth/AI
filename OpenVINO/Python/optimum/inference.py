from optimum.intel import OVModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from openvino.runtime import Core

model = OVModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad-ov-fp32", compile=False)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
if "GPU" in Core().available_devices:
    model.to("GPU")
    model.compile()
ov_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device_map="auto")
result = ov_pipe("What is OpenVINO?", "OpenVINO is a framework for deep learning inference optimization")

print(result)