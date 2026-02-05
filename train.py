from ultralytics import YOLO

def train_yolo_sequential():
    # ==========================
    # 第一個實驗：MNB 數據集
    # ==========================
    print("--- 開始訓練 MNB 數據集 ---")
    model_1 = YOLO('yolo26m.pt')  # 每次訓練前重新載入純淨模型
    
    model_1.train(
        data='data_mnb.yaml',
        epochs=100,
        imgsz=640,
        device='0',
        batch=16,
        name='yolo26_mnb_medium'   # 給予獨特的名稱
    )
    # 釋放記憶體 (可選，但在顯存吃緊時有幫助)
    del model_1 

    # ==========================
    # 第二個實驗：Mosaic 數據集
    # ==========================
    print("--- 開始訓練 Mosaic 數據集 ---")
    model_2 = YOLO('yolo26m.pt')  # ⚠️ 關鍵：重新載入原始模型
    
    model_2.train(
        data='data_mosaic.yaml',
        epochs=100,
        imgsz=640,
        device='0',
        batch=16,
        name='yolo26_mosaic_medium' # 給予獨特的名稱
    )

    print("所有訓練任務已完成！")

if __name__ == '__main__':
    train_yolo_sequential()