"""Gradio demo: upload a photo, get P(Dog) and a Grad-CAM heatmap.

Run locally:            python app.py --model runs/effnet/model.keras
On Hugging Face Spaces: put model.keras next to this file; it is picked up automatically.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import keras
import numpy as np

from catdog.gradcam import gradcam, overlay


def build_app(model_path):
    model = keras.models.load_model(model_path)
    size = tuple(model.input_shape[1:3])

    def classify(image):
        img = np.asarray(image.convert("RGB").resize(size), dtype=np.float32)
        heat, p_dog = gradcam(model, img)
        return {"Dog": p_dog, "Cat": 1 - p_dog}, overlay(img, heat)

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Photo of a cat or dog"),
        outputs=[gr.Label(label="Prediction"), gr.Image(label="Where the model looked (Grad-CAM)")],
        title="Cats vs Dogs",
        description="EfficientNetB0 fine-tuned on 25k photos. The heatmap shows which regions drove the prediction.",
        flagging_mode="never",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "model.keras"))
    build_app(parser.parse_args().model).launch()
