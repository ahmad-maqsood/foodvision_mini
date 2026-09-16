
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as Timer

class_names = ['pizza', 'steak', 'sushi']

effnetb2, effnetb2_transforms = create_effnetb2_model()

effnetb2.load_state_dict(torch.load(f='pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth',
                         map_location=torch.device('cpu')))


def predict(img):
    start_time = timer()
    
    img = effnetb2_transforms(img).unsqueeze(0)
    
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time

title = 'FoodVision Mini 🍕🥩🍣'
description = 'A pretrained efficientnetb2 model that is modified and optimized to classify the images of pizza, steak, and sushi.'

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=3, label='Predictions'),
                             gr.Number(label='Prediction Time (s)')],
                    examples=example_list,
                    title=title,
                    description=description)
demo.launch()
