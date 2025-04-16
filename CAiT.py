from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Initialize chat history
chat_history_ids = None

print("🐾 Cleo the CatBot is online. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Cleo: *licks paw and curls up for a nap*")
        break

    # Refined cat-like prompt
    start_prompt = ("The following is a conversation with Cleo, a sassy, smart, sarcastic, playful cat who talks like a human but acts like and is actually a cat. Use words like hooman and meow while "
                    "communicating\nHuman: Hello!\nCleo: Well hello, hooman. Have you brought treats or just questions today?\n")

    chat_history_ids = tokenizer.encode(start_prompt, return_tensors='pt')

    # Encode the user input and create an attention mask
    new_input_ids = tokenizer.encode(start_prompt + tokenizer.eos_token, return_tensors='pt')
    attention_mask = torch.ones(new_input_ids.shape, device=new_input_ids.device)

    # Generate response with attention mask and sampling enabled
    chat_history_ids = model.generate(
        new_input_ids,
        attention_mask=attention_mask,  # Pass the attention mask
        do_sample=True,  # Enable sampling mode to use temperature
        max_length=150,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        top_k=50
    )

    # Decode the response and remove the user's input part
    response = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    response = response[len(start_prompt):].strip()  # Strip the prompt from the response

    print(f"Cleo: {response}")