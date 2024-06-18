from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def chatbot_reply(input_text):
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize input text
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # Generate response
    with torch.no_grad():
        # Set chatbot response parameters
        chatbot_output = model.generate(
            input_ids,
            max_length=100,  # Adjust max length of the response
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,  # Adjust top_k sampling parameter
            top_p=0.95,  # Adjust top_p sampling parameter
            temperature=0.9  # Adjust temperature parameter
        )

    # Decode the generated response and remove special tokens
    chatbot_reply = tokenizer.decode(chatbot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return chatbot_reply
