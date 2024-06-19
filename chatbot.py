from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)


def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    generated_output = model.generate(**inputs)
    response = tokenizer.batch_decode(generated_output, skip_special_tokens=True)[0]
    return response
