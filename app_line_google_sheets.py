import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import pyocr
import pyocr.builders
import re
import sys
import time
import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import TextSendMessage

# 引入 Google Sheets API 模組
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# 從環境變數獲取 LINE Bot Token 和 Secret
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')  # LINE access token
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')  # LINE secret

# 從環境變數獲取 Google Sheets API 金鑰
GOOGLE_SHEET_CREDENTIALS_FILE = os.getenv('GOOGLE_SHEET_CREDENTIALS_FILE')  # Google Sheets API credential file

# 初始化 OCR 工具
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("❌ No OCR tool found")
    sys.exit(1)
tool = tools[0]

# 用來偵測車牌的 Haar 分類器
detector = cv2.CascadeClassifier('haar_carplate.xml')

# 讀取 Google Sheets 的車輛資料
def load_vehicle_data_from_sheets(spreadsheet_id, range_name):
    # 授權
    credentials = Credentials.from_service_account_file(
        GOOGLE_SHEET_CREDENTIALS_FILE, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build('sheets', 'v4', credentials=credentials)

    # 讀取資料
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    if not values:
        print("No data found.")
        return None
    else:
        # 將資料轉換成 DataFrame 格式
        import pandas as pd
        df = pd.DataFrame(values[1:], columns=values[0])
        return df

# 查詢車輛資訊
def get_vehicle_info(plate_number, vehicle_data):
    vehicle_info = vehicle_data[vehicle_data['車牌號碼'] == plate_number]
    if not vehicle_info.empty:
        return vehicle_info.iloc[0]
    else:
        return None

def remove_noise_and_segment(thresh):
    contours1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours1[0]
    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if x >= 2 and x <= 125 and w >= 5 and w <= 35 and h >= 20 and h < 40:
            letter_image_regions.append((x, y, w, h))
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    return letter_image_regions

def extract_characters(thresh, letterlist):
    real_shape = []
    for i, box in enumerate(letterlist):
        x, y, w, h = box
        bg = thresh[y:y+h, x:x+w]
        real_shape.append(bg)
    return real_shape

def reconstruct_plate_image(thresh, letterlist, real_shape):
    newH, newW = thresh.shape
    space = 10
    bg = np.zeros((newH + space*2, newW + space*2 + 20, 1), np.uint8)
    bg.fill(0)
    for i, letter in enumerate(real_shape):
        h = letter.shape[0]
        w = letter.shape[1]
        x = letterlist[i][0]
        y = letterlist[i][1]
        for row in range(h):
            for col in range(w):
                bg[space + y + row][space + x + col + i*3] = letter[row][col]
    _, bg = cv2.threshold(bg, 127, 255, cv2.THRESH_BINARY_INV)
    return bg

def ocr_recognition(image_path='result.jpg'):
    result = tool.image_to_string(
        Image.open(image_path),
        builder=pyocr.builders.TextBuilder()
    )
    txt = result.replace("!", "1")
    real_txt = re.findall(r'[A-Z]+|[\d]+', txt)
    txt_plate = "".join(real_txt)
    return txt, txt_plate

# LINE Bot 設定
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
your_user_id = os.getenv('LINE_USER_ID')  # 從環境變數獲取 LINE 使用者 ID

# Flask 初始化
app = Flask(__name__)

# 接收 LINE Webhook 的端點
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 設定中文字型
font_path = "C:/Windows/Fonts/msjh.ttc"
font = ImageFont.truetype(font_path, 36)

# 啟動攝影機
cap = cv2.VideoCapture(0)

# 車主資訊記錄
last_detection_time = 0
car_owner = ""
detection_in_progress = False

print("📷 開始從攝影機擷取畫面進行車牌辨識...（按 q 結束）")

# 讀取車輛資料 Google Sheets ID 和範圍
spreadsheet_id = '1gM7sqVtmsB880hdUFGdL3gltlGtCWPFRLSOdTGgyZsM'
range_name = '工作表1!A1:B100' # 假設資料範圍是從 A1 到 D 欄

# 讀取 Google Sheets 的車輛資料
vehicle_data = load_vehicle_data_from_sheets(spreadsheet_id, range_name)

# 主迴圈
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if detection_in_progress and (time.time() - last_detection_time < 10):
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((50, 50), f"車主：{car_owner}", font=font, fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("原始攝影畫面", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    signs = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    result_img = None
    for (x, y, w, h) in signs:
        plate = frame[y:y+h, x:x+w]
        plate_pil = Image.fromarray(plate).resize((140, 40), Image.ANTIALIAS)
        plate_gray = np.array(plate_pil.convert('L'))
        _, thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY_INV)

        letterlist = remove_noise_and_segment(thresh)
        if len(letterlist) == 0:
            continue
        real_shape = extract_characters(thresh, letterlist)
        result_img = reconstruct_plate_image(thresh, letterlist, real_shape)
        cv2.imwrite('result.jpg', result_img)
        break

    if result_img is not None:
        txt, txt_plate = ocr_recognition('result.jpg')
        print("🔤 原始辨識：", txt)
        print("✅ 優化結果：", txt_plate)

        vehicle_info = get_vehicle_info(txt_plate, vehicle_data)
        if vehicle_info is not None:
            car_owner = vehicle_info['車主姓名']
            last_detection_time = time.time()
            detection_in_progress = True

            # 發送訊息到 LINE
            line_bot_api.push_message(your_user_id, TextSendMessage(text=f"🚗 偵測到車主：{car_owner}\n🔢 車牌號碼：{txt_plate}"))

    cv2.imshow("原始攝影畫面", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
