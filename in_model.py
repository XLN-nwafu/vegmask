from flask import Flask, jsonify, request, send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import io
from demo.predictor import VisualizationDemo
from detectron2.config import get_cfg

app = Flask(__name__)

import requests

def download_model(model_url):
    response = requests.get(model_url)
    if response.status_code == 200:
        with open('model_final.pth', 'wb') as f:
            f.write(response.content)
        return 'model_final.pth'
    else:
        return None
# 预测接口
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file']
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)

    model = download_model("cloud://prod-6glf7ij51e56f201.7072-prod-6glf7ij51e56f201-1329362844/model_final.pth")

    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/COCO-InstanceSegmentation/maskrcnn_mobilenetv3l_FPN.yaml")
    opts = ["MODEL.WEIGHTS", model]
    cfg.merge_from_list(opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.freeze()

    demo = VisualizationDemo(cfg)
    predictions, visualized_output = demo.run_on_image(image)
    pred = visualized_output.get_image()


    return send_file(pred, mimetype='image/png')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)