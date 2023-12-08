import flask
from PIL import Image
import io
from docgcn_inference import inference_main
from flask import Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Hello World! Basic check works!!"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        result = inference_main(image)

        data["response"] = result
        data["success"] = True
    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8001)

# Command to run
# python app.py