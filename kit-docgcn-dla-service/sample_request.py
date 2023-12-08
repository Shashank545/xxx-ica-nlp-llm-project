import argparse
import requests
import os
import sys
import time
import json
from pathlib import Path
from PIL import ImageDraw, ImageFont, Image

API_URL = 'http://127.0.0.1:8001/predict'

start_inf_time = time.time()

def predict_result(image_path):
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    res = requests.post(API_URL, files=payload)
    print(res)
    r = res.json()

    return r


# Define default values
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, help='img')

args = vars(parser.parse_args())
print(args['img'])

# img_path = sys.argv[1]
img_path = args['img']

basepath = "/".join(img_path.split("/")[:-1])
file = img_path.split("/")[-1].split(".")[0]

# filepath = Path("data/inference_data/83443897_pred.json")
# overlayimgpath = Path("data/inference_data/83443897_dla.png")

filepath = basepath + "/dla_results/" + file + "_pred_new.json"
print(filepath)
overlayimgpath = basepath + "/dla_results/" + file + "_dla_new.png"
print(overlayimgpath)

if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
    os.remove(filepath)

if os.path.isfile(overlayimgpath) and os.access(overlayimgpath, os.R_OK):
    os.remove(overlayimgpath)


print("Checking results for {}".format(img_path))
result = predict_result(img_path)

with open(filepath, "w") as foo:
    json.dump(result["response"], foo, ensure_ascii=False, indent=4)

# with open(filepath, "r") as foo:
#     res_data = json.load(foo)
res_data = result["response"]


inf_image = Image.open(img_path)
inf_image = inf_image.convert("RGBA")
draw = ImageDraw.Draw(inf_image)
font = ImageFont.load_default()
iob_to_label = {3:'question', 0: 'answer', 2: 'other', 1: 'header'}
label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

predictions = [res["label_predictions"] for res in res_data]
bboxes = [res["bbox"] for res in res_data]

for prediction, box in zip(predictions, bboxes):
    predicted_label = iob_to_label[prediction]
    draw.rectangle(box, outline=label2color[predicted_label])
    draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)



inf_image.save(overlayimgpath)
# print(result)
print("<Results successfully retrieved!>")
stop_inf_time = time.time()
print("--- %s seconds ---" % (stop_inf_time - start_inf_time))


# command to run
# python sample_request.py --img data/general_images/83443897.png