#攝影機
import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown
import os
import cv2
from threading import Thread
from queue import Queue
import time

# 從Google Drive下載模型（請替換為實際的檔案ID）
model_id = '1vhAAyel66fPNCa_zqFKSgswbC2hNANef'
output = 'model_best.h5'
if not os.path.exists(output):
    gdown.download(f'https://drive.google.com/uc?id={model_id}', output, quiet=False)

# 載入模型
model = load_model(output)

def detect_license_plate(image):
    # 將圖片轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 高斯模糊處理
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 自適應閾值處理
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 形態學操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # 邊緣檢測
    edges = cv2.Canny(morph, 30, 200)

    # 尋找輪廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 篩選可能的車牌區域
    license_plate = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) >= 4 and len(approx) <= 8:
                (x, y, w, h) = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)

                # 車牌的寬高比和面積條件
                if 2.0 <= aspect_ratio <= 5.0 and area > max_area:
                    max_area = area
                    # 提取稍大一點的區域，避免切到邊緣
                    padding = 5
                    y_start = max(y - padding, 0)
                    y_end = min(y + h + padding, image.shape[0])
                    x_start = max(x - padding, 0)
                    x_end = min(x + w + padding, image.shape[1])
                    license_plate = image[y_start:y_end, x_start:x_end]

    if license_plate is not None:
        # 對車牌區域進行進一步的圖像增強
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        license_plate_eq = cv2.equalizeHist(license_plate_gray)
        license_plate = cv2.cvtColor(license_plate_eq, cv2.COLOR_GRAY2BGR)

    return license_plate

def preprocess_image(image):
    # 檢測車牌區域
    plate = detect_license_plate(image)
    if plate is None:
        return None

    # 預處理車牌圖片
    img = cv2.resize(plate, (224, 224))  # 調整大小

    # 轉換為灰度圖並進行圖像增強
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)

    # 二值化處理
    _, img_thresh = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 去除噪點
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_denoise = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    # 正規化並擴展維度
    img_norm = img_denoise.astype(np.float32) / 255.0
    img_final = np.expand_dims(img_norm, axis=0)

    return img_final

def predict_image(image):
    # 預處理圖片
    processed_img = preprocess_image(image)
    if processed_img is None:
        return {"未檢測到車牌": 1.0}

    # 進行預測
    prediction = model.predict(processed_img)

    # 解碼預測結果
    license_plate = decode_predictions(prediction[0])

    # 計算置信度分數
    confidence_scores = np.max(prediction[0], axis=-1)
    mean_confidence = float(np.mean(confidence_scores))

    # 驗證車牌格式
    if validate_license_plate(license_plate):
        return {f'車牌號碼: {license_plate}': mean_confidence}
    else:
        return {"無效的車牌號碼": mean_confidence}

def decode_predictions(pred):
    # 假設模型輸出為每個字符的機率分布
    # 這裡需要根據實際模型輸出格式進行調整
    char_list = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
    result = ''

    # 對每個字符位置選擇最高機率的字符
    for char_prob in pred:
        if len(char_prob) == len(char_list):
            char_idx = np.argmax(char_prob)
            result += char_list[char_idx]

    return result

def validate_license_plate(plate):
    # 驗證車牌格式是否符合規範
    # 這裡需要根據實際車牌格式規則進行調整
    if not plate:
        return False

    # 檢查長度（假設台灣車牌為7位）
    if len(plate) != 7:
        return False

    # 檢查格式（假設格式為：2英文+5數字）
    if not (plate[:2].isalpha() and plate[2:].isdigit()):
        return False

    return True

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.queue = Queue(maxsize=3)

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    return
                if not self.queue.full():
                    self.queue.put(frame)

    def read(self):
        return self.queue.get() if not self.queue.empty() else None

    def stop(self):
        self.stopped = True
        self.stream.release()

def process_camera_feed():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)  # 等待攝影機初始化

    last_prediction_time = 0
    prediction_interval = 1.0  # 每秒預測一次

    while True:
        frame = vs.read()
        if frame is None:
            continue

        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            # 進行車牌辨識
            result = predict_image(frame)
            last_prediction_time = current_time

        # 在影像上顯示辨識結果
        for text, conf in result.items():
            cv2.putText(frame, f"{text}: {conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, result

# 創建 Gradio 界面
iface = gr.Interface(
    fn=process_camera_feed,
    inputs=None,
    outputs=[gr.Image(label="攝影機畫面"), gr.Label(num_top_classes=1)],
    title="即時車牌辨識系統",
    description="使用攝影機進行即時車牌辨識。",
    live=True,
    refresh_per_second=10
)

# 啟動應用
if __name__ == "__main__":
    iface.launch()
