from optimum.intel import OVModelForQuestionAnswering
from transformers import AutoTokenizer

# Load PyTorch model from the Hub and export to OpenVINO in the background
model = OVModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad", export=True)
# Save the converted model to a local directory
model.save_pretrained("distilbert-base-uncased-distilled-squad-ov-fp32")

# Load tokenizer from a PyTorch model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
# Save tokenizer
tokenizer.save_pretrained("distilbert-base-uncased-distilled-squad-ov-fp32")

# Load the OpenVINO model directly from the directory
model = OVModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad-ov-fp32")
# Load the tokenizer directly from the directory
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad-ov-fp32")
