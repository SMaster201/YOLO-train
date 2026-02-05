from ultralytics import YOLO
import sys
import torch
import shutil
import numpy as np
from pathlib import Path

# ================= 設定區 =================
# 1. 原始 FP32 模型路徑
SOURCE_MODEL = 'runs/detect/yolo26_mosaic_medium/weights/MOSAIC.pt'

# 2. 資料集設定檔 (INT8 量化必須使用此檔案讀取圖片來校準權重)
DATA_YAML = 'data_mosaic.yaml'

# 3. 圖片輸入尺寸
IMGSZ = 640
# ========================================

def check_numpy_version():
    """檢查 NumPy 版本，避免 2.x 版本導致導出失敗"""
    v = np.__version__
    print(f"目前 NumPy 版本: {v}")
    if v.startswith('2'):
        print("⚠️ 警告: 偵測到 NumPy 2.0+。")
        print("   這通常會導致 TFLite 和 OpenVINO 導出失敗 (錯誤: module 'numpy' has no attribute...)。")
        print("   解決方案: 請執行 `pip install \"numpy<2.0\"` 降級 NumPy。")
        print("-" * 50)

def export_model(model, format_type, suffix=None, **kwargs):
    """封裝導出函式，包含錯誤處理與自動重新命名"""
    label = kwargs.get('label', format_type)
    print(f"\n[{label}] 正在導出...")
    
    # 移除自定義參數以免干擾 export
    if 'label' in kwargs: del kwargs['label']
    
    try:
        # 執行導出
        path = model.export(format=format_type, imgsz=IMGSZ, data=DATA_YAML, **kwargs)
        
        # 處理路徑物件 (有時回傳是字串)
        if isinstance(path, str):
            path = Path(path)
        elif isinstance(path, list):
             path = Path(path[0]) # 有些格式可能回傳 list
        
        # 自動重新命名 (避免 FP16 與 INT8 檔案覆蓋)
        final_path = path
        if suffix and path.exists():
            new_name = path.stem + suffix + path.suffix
            new_path = path.parent / new_name
            # 如果目標檔案已存在，先刪除
            if new_path.exists():
                new_path.unlink()
            path.rename(new_path)
            final_path = new_path
            
        print(f"✅ {label} 成功: {final_path}")
        return (label, str(final_path))
        
    except Exception as e:
        print(f"❌ {label} 失敗: {e}")
        return None

def main():
    check_numpy_version()

    model_path = Path(SOURCE_MODEL)
    if not model_path.exists():
        print(f"錯誤: 找不到模型檔案 {SOURCE_MODEL}")
        return

    print(f"載入模型: {model_path.name}")
    try:
        model = YOLO(SOURCE_MODEL)
    except Exception as e:
        print(f"無法載入 YOLO 模型: {e}")
        return

    print("\n=== 開始多重精度量化導出流程 ===")
    print("注意: INT8 量化過程需要讀取資料集進行校準 (Calibration)，請耐心等待。")

    output_files = []

    # =========================================================
    # 1. FP16 (半精度)
    # =========================================================
    # TensorRT FP16 (NVIDIA GPU)
    if torch.cuda.is_available():
        res = export_model(model, 'engine', half=True, device=0, 
                          suffix='_fp16', label='[FP16] TensorRT Engine')
        if res: output_files.append(res)
    else:
        print("\n⚠️ 略過 TensorRT (需要 NVIDIA GPU)")

    # =========================================================
    # 2. INT8 (8-bit 整數)
    # ========================================================= 
    # TensorRT INT8 (需要 GPU)
    if torch.cuda.is_available():
        print("\n[注意] 正在嘗試導出 TensorRT INT8，這可能需要較長時間...")
        res = export_model(model, 'engine', int8=True, device=0, 
                          suffix='_int8', label='[INT8] TensorRT Engine')
        if res: output_files.append(res)

    # =========================================================
    # 3. 關於 FP8 與 INT4 的說明
    # =========================================================
    print("\n" + "-"*30)
    print("ℹ️ 關於 FP8 與 INT4 格式支援:")
    print("目前 YOLO 官方 `export` 指令尚未直接支援 FP8/INT4 參數。")
    print("1. [INT4]: 通常需使用 OpenVINO NNCF 的 `weight_compression` 工具進行後處理，")
    print("          或使用 GPTQ/AWQ 針對 LLM 的工具，但對 YOLO 檢測模型支援較少。")
    print("2. [FP8]: 僅支援 NVIDIA Ada Lovelace (RTX 40系列) 或 Hopper (H100) 架構，")
    print("          且需透過 TensorRT 底層 API 轉換，目前無法透過單行 Python 指令完成。")
    print("-"*30)

    # =========================================================
    # 總結
    # =========================================================
    print("\n" + "="*50)
    print("導出結果總結 (請將這些路徑用於 batch_eval_yolo.py):")
    for fmt, p in output_files:
        print(f"{fmt: <25}: {p}")
    print("="*50)

if __name__ == '__main__':
    main()