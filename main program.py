import os
import cv2
import numpy as np
import csv
import time
from matplotlib import pyplot as plt
from datetime import datetime

# 配置變數
extension = '.jpg'
patternsPath = 'solid_patterns/'
databasePath = 'car_owners.csv'

def extract_and_recognize_chars(plate_image, patterns_path):
    # 獲取字符輪廓
    contours, hierarchy = cv2.findContours(plate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    letters = []

    # 過濾和提取字符
    for contour in contours:
        rect = cv2.boundingRect(contour)
        if (rect[3] > (rect[2] * 1.5)) and (rect[3] < (rect[2] * 3.5) and (rect[2] > 10)):
            letters.append(rect)

    # 排序並提取字符圖像
    letter_images = []
    for letter in sorted(letters, key=lambda s: s[0]):
        x, y, w, h = letter
        letter_images.append(plate_image[y:y+h, x:x+w])

    # 識別字符
    results = []
    for letter_image in letter_images:
        best_score = []
        patterns = os.listdir(patterns_path)

        for filename in patterns:
            pattern_img = cv2.imdecode(np.fromfile(os.path.join(patterns_path, filename), dtype=np.uint8), 1)
            pattern_gray = cv2.cvtColor(pattern_img, cv2.COLOR_RGB2GRAY)
            _, pattern_binary = cv2.threshold(pattern_gray, 0, 255, cv2.THRESH_OTSU)
            pattern_resized = cv2.resize(pattern_binary, (letter_image.shape[1], letter_image.shape[0]))

            score = cv2.matchTemplate(letter_image, pattern_resized, cv2.TM_CCOEFF)[0][0]
            best_score.append(score)

        if best_score:
            best_match = patterns[best_score.index(max(best_score))]
            results.append(best_match.replace(extension, ''))

    return ''.join(results) if results else None

def process_frame(frame, patterns_path):
    try:
        # 轉換顏色空間
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 車牌檢測
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        edges = cv2.Canny(blurred, 170, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 尋找可能的車牌區域
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)

                # 檢查區域大小是否合理（避免誤檢）
                if w < 60 or h < 20:  # 太小的區域跳過
                    continue

                # 檢查長寬比是否符合車牌特徵
                aspect_ratio = float(w) / h
                if not (2.0 <= aspect_ratio <= 5.5):  # 車牌的典型長寬比範圍
                    continue

                # 繪製檢測框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 提取車牌區域
                plate_region = frame_bgr[y:y + h, x:x + w]
                if plate_region.size == 0:
                    continue

                # 處理車牌圖像
                _, plate_binary = cv2.threshold(
                    cv2.cvtColor(
                        cv2.GaussianBlur(plate_region, (3, 3), 0),
                        cv2.COLOR_RGB2GRAY
                    ),
                    0, 255, cv2.THRESH_OTSU
                )

                # 提取字符
                plate_number = extract_and_recognize_chars(plate_binary, patterns_path)
                if plate_number:
                    # 在畫面上顯示車牌號碼
                    cv2.putText(
                        frame,
                        f"車牌: {plate_number}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    return frame, plate_number

        return frame, None
    except Exception as e:
        print(f"處理幀時發生錯誤: {str(e)}")
        return frame, None

def start_camera_recognition():
    # 添加視窗設置
    cv2.namedWindow('License Plate Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('License Plate Recognition', 1280, 720)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("無法開啟攝像頭")
        return

    last_recognition_time = 0
    recognition_cooldown = 2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝像頭畫面")
                break

            # 添加幀率顯示
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            current_time = time.time()

            if current_time - last_recognition_time >= recognition_cooldown:
                processed_frame, plate_number = process_frame(frame, patternsPath)

                if plate_number:
                    last_recognition_time = current_time
                    try:
                        with open(databasePath, mode="r", encoding="utf-8-sig") as file:
                            reader = csv.DictReader(file)
                            for row in reader:
                                if row["車牌號碼"].strip() == plate_number:
                                    info = f"""
時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
車牌號碼: {plate_number}
車主姓名: {row['車主姓名']}
{'='*30}"""
                                    print(info)
                                    break
                    except FileNotFoundError:
                        print("無法讀取資料庫文件")
            else:
                processed_frame = frame

            cv2.imshow('License Plate Recognition', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_recognition()