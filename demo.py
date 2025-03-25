from transformers import AutoTokenizer
from utils.models import Baseline
import gradio as gr
import torch

label2id = {'Anxiety': 0,
    'Normal': 1,
    'Depression': 2,
    'Suicidal': 3,
    'Stress': 4,
    'Bipolar': 5,
    'Personality disorder': 6
}


def use_model(text):
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Meta-Llama-3.1-8B-Instruct')
    model = AutoTokenizer.from_pretrained("goodevening13/model_hse_hw")
    tokens = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")['input_ids']
    logits = model(tokens)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).cpu()
    return label2id[preds[0]]


demo = gr.Interface(
    fn=lambda text: use_model(text),
    inputs=gr.Textbox(placeholder="Введите текст..."),
    outputs="label",
    title="Классификация текста"
)

demo.launch(server_port=7860)