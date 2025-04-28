from flask import Flask, render_template, Response
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# 讀取車牌與車主資料
license_data = pd.read_csv('license_plate_owners.csv')

# 載入訓練好的模型
model = load_model('model_best.h5')

# 讀取模型預測車牌號碼的功能
def decode_predictions(pred):
    char_list = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
    result = ''
    for char_prob in pred:
        if len(char_prob) == len(char_list):
            char_idx = np.argmax(char_prob)
            result += char_list[char_idx]
    return result

def preprocess_image(image):
    # 這裡省略車牌偵測與預處理過程，假設已經偵測到車牌區域
    # 假設車牌區域已經被裁切為 224x224
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

def predict_license_plate(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    plate = decode_predictions(prediction[0])
    return plate

# 查詢車主資料
def get_owner_by_plate(plate):
    match = license_data[license_data['license_plate'] == plate]
    if not match.empty:
        return match['owner_name'].values[0]
    return "未找到車主"

# 開啟攝影機並顯示即時畫面
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        plate = predict_license_plate(frame)
        owner = get_owner_by_plate(plate)

        # 在畫面上顯示車主名稱
        cv2.putText(frame, f'車牌: {plate}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'車主: {owner}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 編碼成 JPEG 發送到網頁
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
