# from flask import Flask, jsonify, request, send_file
import cv2
# import run_model
# import numpy as np
# import base64
# app = Flask(__name__)

# @app.route("/check")
# def check():
#     return jsonify({"received":"200"})

# @app.route('/api', methods=["POST"])
# def gateway():
#     file = request.files['image']
#     npimg = np.fromfile(file, np.uint8)
#     img = cv2.cvtColor(cv2.imdecode(npimg, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
#     enhanced_image = run_model.run_depd(img,1536,2048,9437184)
#     retval, buffer = cv2.imencode('.png', enhanced_image)
#     return jsonify({'imagekey':base64.b64encode(buffer)})

# if __name__=="__main__":
#     app.run(host="10.0.76.50")

import os
path = os.listdir("test_photos/landscape")
for i in path:
    # img = cv2.cvtColor(cv2.imdecode('test_photos/landscape/'+i, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
    print(i)
    img = cv2.imread('test_photos/landscape/'+i)
    print(img.shape)
    img = cv2.resize(img,(2048,1536))
    print((img.shape))
    # enhanced_image = run_model.run_depd(img,1536,2048,9437184)