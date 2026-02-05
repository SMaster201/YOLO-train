import json
import os
from pathlib import Path

def convert_coco_json_to_yolo(json_file, output_label_dir):
    # 建立輸出資料夾
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 讀取 JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 建立圖片 ID 對應檔名與尺寸的字典
    images = {img['id']: img for img in data['images']}
    
    # 處理每一個標註
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_info = images.get(img_id)
        
        if not img_info:
            continue
            
        # 取得圖片尺寸
        img_w = img_info['width']
        img_h = img_info['height']
        
        # COCO bbox: [x_min, y_min, width, height]
        x_min, y_min, w, h = ann['bbox']
        
        # 轉換為 YOLO 格式: [x_center, y_center, width, height] (全部歸一化 0~1)
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # 類別 ID (直接使用 json 中的 id，假設是 0-10)
        class_id = ann['category_id']
        
        # 準備寫入的字串
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        
        # 檔名處理：將圖片副檔名換成 .txt
        img_filename = img_info['file_name']
        txt_filename = os.path.splitext(img_filename)[0] + ".txt"
        
        # 寫入檔案 (附加模式，因為一張圖可能有多個物件)
        output_path = os.path.join(output_label_dir, txt_filename)
        with open(output_path, 'a') as out_f:
            out_f.write(yolo_line)

    print(f"轉換完成！標註檔已存入: {output_label_dir}")

# --- 使用設定 ---
# 請針對您的三個資料夾分別執行三次，或修改下方路徑
if __name__ == "__main__":
    # 範例：轉換訓練集
    # 假設您的 JSON 在 datasets/train/_annotations.coco.json
    # 想要輸出到 datasets/labels/train
    
    # 1. 轉換 Train
    if os.path.exists('nmosv8_mnb_bmp/train/_annotations.coco.json'):
        convert_coco_json_to_yolo(
            json_file='nmosv8_mnb_bmp/train/_annotations.coco.json', 
            output_label_dir='nmosv8_mnb_bmp/labels/train'
        )
    
    # 2. 轉換 Valid
    if os.path.exists('nmosv8_mnb_bmp/valid/_annotations.coco.json'):
        convert_coco_json_to_yolo(
            json_file='nmosv8_mnb_bmp/valid/_annotations.coco.json', 
            output_label_dir='nmosv8_mnb_bmp/labels/val'
        )

    # 3. 轉換 Test (如果有)
    if os.path.exists('nmosv8_mnb_bmp/test/_annotations.coco.json'):
        convert_coco_json_to_yolo(
            json_file='nmosv8_mnb_bmp/test/_annotations.coco.json', 
            output_label_dir='nmosv8_mnb_bmp/labels/test'
        )