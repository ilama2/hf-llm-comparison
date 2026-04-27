def generate_response(prompt, max_new_tokens=150, tokenizer=None, model=None):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    is_encoder_decoder = model.config.is_encoder_decoder

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not is_encoder_decoder,   # ✔️ مهم
            temperature=0.7 if not is_encoder_decoder else 0,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = full_output[len(prompt):].strip()

    return response

def explain_concept(concept, tokenizer, model):
    prompt = f"""
    You are an AI tutor.
    Explain clearly in simple terms.
    Do NOT include unrelated topics.

    Concept: {concept}
    """
    return generate_response(prompt=prompt, tokenizer=tokenizer, model=model)

def summarize_text(text,tokenizer, model):
    prompt = f"""
    Summarize the following text into 3 clear bullet points.
    Only output the summary.
    Do not explain anything.
    Text:
    {text}

    Summary:
    """
    return generate_response(prompt, tokenizer=tokenizer, model=model)

def answer_question(context, question, tokenizer, model):
    prompt = f"""
    Answer ONLY using the given context.
    If the answer is not in the context, say: "Not found".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return generate_response(prompt, tokenizer=tokenizer, model=model)

