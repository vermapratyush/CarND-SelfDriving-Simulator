import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2
import math

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def get_img(img):
    img = img[55:135, :]
    img = cv2.resize(img, (64, 64))
    img = img / 255.0 - 0.5
    return img

def flip(img, y):
    return -1 * float(y), cv2.flip(img, 1)

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
#   image_array = cv2.resize(image_array, (200, 100))
    transformed_image_array = get_img(image_array)
    
    print(transformed_image_array.shape)
    
    transformed_image_array = np.reshape(transformed_image_array, (1, transformed_image_array.shape[0], transformed_image_array.shape[1], transformed_image_array.shape[2]))

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
# Add the preprocessing step
#    image_array = preprocess_image(image_array)

#    transformed_image_array = image_array[None, :, :, :] 
    print(transformed_image_array.shape)
    steering_angle = model.predict(transformed_image_array, batch_size=1)[0][0]
    
    print(steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.1
    throttle = 0.17 / math.exp(2*math.fabs(steering_angle))
    throttle = 0.3
#    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
#         model = model_from_json(json.loads(jfile.read()))
        #
        # instead.
        x=jfile.read()
        model = model_from_json(x)
        print(x)
        


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    print(weights_file)
    model.load_weights(weights_file)
    

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)