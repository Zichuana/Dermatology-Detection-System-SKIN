from flask import Flask, jsonify, render_template, request
from datetime import timedelta
import torch
from model import resnet34
import torchvision.transforms as transforms
import torch.utils.data.dataloader as Data
from PIL import Image
import io
import json

app = Flask(__name__, static_url_path="/")

# 自动重载模板文件
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
# 设置静态文件缓存过期时间
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
weights_path = './ResNet34_0.613.pth'
model = resnet34(num_classes=8).to(device)
with open("./class_indices.json", 'r', encoding='GBK') as f:
    class_indict = json.load(f)
model.load_state_dict(torch.load(weights_path, map_location=device))


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    try:
        skin_str = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        tensor = transform_image(image_bytes=image_bytes)
        output = torch.softmax(model.forward(tensor.to(device)).squeeze(), dim=0)
        pre = output.detach().cpu().numpy()
        template = "种类:{:<15} 可能性:{:.3f}"
        index_pre = [(class_indict[str(skin_str[int(index)])], float(p)) for index, p in enumerate(pre)]
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)  # json格式传至前端


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1234)