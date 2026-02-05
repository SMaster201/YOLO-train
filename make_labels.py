import json
import os
import glob

# 設定您的資料集路徑 (根據您的 Log 修改)
# 建議檢查路徑是否與您電腦完全一致
base_path = r"D:\專題\yolo\nmosv8_mnb_bmp"
dirs = ['train', 'valid', 'test']

def convert_coco_to_yolo(folder_name):
    dir_path = os.path.join(base_path, folder_name)
    json_path = os.path.join(dir_path, "_annotations.coco.json")
    
    if not os.path.exists(json_path):
        print(f"跳過: 找不到 {json_path}")
        return

    print(f"正在處理: {folder_name} ...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立圖片ID對應表
    images = {item['id']: item for item in data['images']}
    
    # 統計轉換數量
    count = 0
    
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in images:
            continue
            
        img_info = images[image_id]
        file_name = img_info['file_name']
        
        # 轉換檔名 .bmp -> .txt
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(dir_path, txt_name)
        
        # 取得尺寸與座標
        img_w = img_info['width']
        img_h = img_info['height']
        x, y, w, h = ann['bbox']
        
        # 歸一化 (XYWH -> 0~1)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        class_id = ann['category_id']
        
        # 寫入 txt (附加模式，因為一張圖可能有多個物件)
        with open(txt_path, 'a') as out_file:
            out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
        count += 1

    print(f"完成！共生成 {count} 個標籤物件在 {dir_path}")

# 執行轉換
for d in dirs:
    convert_coco_to_yolo(d)