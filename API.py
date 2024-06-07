from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the YOLOv8 model
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

#load the labels
with open("models/classes.txt", "r") as f:
    labels = f.read().strip().split("\n")

# Function to preprocess image for YOLOv8
def preprocess_image(image):
    resized_image = cv2.resize(image, (640, 640))
    resized_image = resized_image / 255.0
    resized_image = resized_image[np.newaxis, ...].astype(np.float16)
    return resized_image

# Function to perform object detection
def detect_objects(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])
    scores = interpreter.get_tensor(output_details[1]['index'])
    classes = interpreter.get_tensor(output_details[2]['index'])

    return boxes, scores, classes

# Route to handle image upload
@app.route('/detect', methods=['POST'])
def detect_food():
    if 'image' not in request.files:
        return jsonify({"error": "No image found in request"}), 400

    image_file = request.files['image']
    image_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    preprocessed_image = preprocess_image(image_np)
    boxes, scores, classes = detect_objects(preprocessed_image)

    # Assuming food class is class 0
    detect_objects = []
    for i in range(len(scores)):
        if scores[i] > 0.1:  # Threshold for detection
            detected_objects.append({
                "box": boxes[i].tolist(),
                "score": float(scores[i]),
                "class": labels[int(classes[i])]
            })

    # You can process detected food here, e.g., get labels, etc.
    return jsonify({"detected_objects": detected_objects}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
