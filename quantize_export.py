"""
YOLO 量化與導出腳本
支援：TensorRT FP16、TensorRT INT8（需校準資料集），可選 ONNX / TFLite / OpenVINO
"""
from pathlib import Path
import sys

import numpy as np
import torch
from ultralytics import YOLO

# ================= 設定區 =================
# 要量化的 .pt 模型路徑列表
MODELS_LIST = [
    "runs/detect/768/yolo26_mosaic_nano/weights/best.pt",
    "runs/detect/768/yolo26_mosaic_small/weights/best.pt",
    "runs/detect/768/yolo26_mosaic_medium/weights/best.pt",
    "runs/detect/768/yolo26_mosaic_large/weights/MOSAIC.pt",
    "runs/detect/768/yolo26_mosaic_xlarge/weights/MOSAIC.pt",
    
    "runs/detect/768/yolo26_mnb_nano/weights/best.pt",
    "runs/detect/768/yolo26_mnb_small/weights/best.pt",
    "runs/detect/768/yolo26_mnb_medium/weights/best.pt",
    "runs/detect/768/yolo26_mnb_large/weights/MNB.pt",
    "runs/detect/768/yolo26_mnb_xlarge/weights/MNB.pt",
]

# 依模型路徑選擇 data.yaml：路徑含 "mnb" 用 data_mnb.yaml，含 "mosaic" 用 data_mosaic.yaml
DATA_YAML_BY_KEYWORD = {
    "mnb": "data_mnb.yaml",
    "mosaic": "data_mosaic.yaml",
}
DATA_YAML_DEFAULT = "data_mosaic.yaml"  # 路徑無法判斷時使用

# 輸入影像尺寸
IMGSZ = 768

# 要導出的格式（可多選）
# 選項: "engine_fp16", "engine_int8", "onnx", "tflite", "openvino"
EXPORT_FORMATS = [
    "engine_fp16",
    "engine_int8",
]

# TensorRT 選項（僅 engine 格式有效）
TRT_DEVICE = 0
TRT_WORKSPACE = 4  # GB
TRT_BATCH = 1

# 導出檔名後綴（避免覆蓋）
SUFFIX_FP16 = "_fp16"
SUFFIX_INT8 = "_int8"
# ========================================


def get_data_yaml_for_model(model_path: str) -> str:
    """依模型路徑關鍵字回傳對應的 data.yaml（mnb -> data_mnb.yaml, mosaic -> data_mosaic.yaml）"""
    path_lower = model_path.lower()
    for keyword, yaml_file in DATA_YAML_BY_KEYWORD.items():
        if keyword in path_lower:
            return yaml_file
    return DATA_YAML_DEFAULT


def check_environment():
    """檢查 NumPy 與 CUDA 環境"""
    print(f"NumPy 版本: {np.__version__}")
    if np.__version__.startswith("2"):
        print("⚠️ NumPy 2.x 可能導致 TFLite/OpenVINO 導出失敗，建議: pip install \"numpy<2\"")
    if torch.cuda.is_available():
        print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 未偵測到 CUDA，將略過 TensorRT 導出")
    print()


def export_one(model, data_yaml: str, fmt_key, **kwargs):
    """
    執行單一格式導出，並依設定加上後綴重新命名。
    data_yaml: 該模型使用的校準/資料集 yaml。
    回傳導出檔案路徑字串，失敗回傳 None。
    """
    label = kwargs.pop("label", fmt_key)
    print(f"  [{label}] 導出中 (data={data_yaml})...")
    fmt = kwargs.pop("format", None)
    suffix = kwargs.pop("suffix", None)
    try:
        out = model.export(format=fmt, imgsz=IMGSZ, data=data_yaml, **kwargs)
        path = Path(out[0] if isinstance(out, list) else out)
        if suffix and path.exists():
            new_path = path.parent / (path.stem + suffix + path.suffix)
            if new_path.exists():
                new_path.unlink()
            path.rename(new_path)
            path = new_path
        print(f"  ✅ {label} 完成: {path}")
        return str(path)
    except Exception as e:
        print(f"  ❌ {label} 失敗: {e}")
        return None


def run_export_for_model(source_pt: str) -> list:
    """對單一 .pt 依 EXPORT_FORMATS 執行所有勾選的導出，回傳成功路徑列表"""
    path = Path(source_pt)
    if not path.exists():
        print(f"跳過（檔案不存在）: {source_pt}")
        return []
    if path.suffix.lower() != ".pt":
        print(f"跳過（非 .pt）: {source_pt}")
        return []

    data_yaml = get_data_yaml_for_model(source_pt)
    print(f"\n{'='*60}")
    print(f"模型: {path.name}  → data: {data_yaml}")
    print("="*60)
    try:
        model = YOLO(str(path))
    except Exception as e:
        print(f"載入失敗: {e}")
        return []

    results = []
    cuda_ok = torch.cuda.is_available()

    if "engine_fp16" in EXPORT_FORMATS and cuda_ok:
        r = export_one(
            model,
            data_yaml,
            "engine_fp16",
            format="engine",
            half=True,
            device=TRT_DEVICE,
            workspace=TRT_WORKSPACE,
            batch=TRT_BATCH,
            suffix=SUFFIX_FP16,
            label="TensorRT FP16",
        )
        if r:
            results.append(r)

    if "engine_int8" in EXPORT_FORMATS and cuda_ok:
        print("  (INT8 校準可能較久...)")
        r = export_one(
            model,
            data_yaml,
            "engine_int8",
            format="engine",
            int8=True,
            device=TRT_DEVICE,
            workspace=TRT_WORKSPACE,
            batch=TRT_BATCH,
            suffix=SUFFIX_INT8,
            label="TensorRT INT8",
        )
        if r:
            results.append(r)

    if "onnx" in EXPORT_FORMATS:
        r = export_one(
            model,
            data_yaml,
            "onnx",
            format="onnx",
            label="ONNX",
        )
        if r:
            results.append(r)

    if "tflite" in EXPORT_FORMATS:
        r = export_one(
            model,
            data_yaml,
            "tflite",
            format="tflite",
            int8=True,
            suffix=SUFFIX_INT8,
            label="TFLite INT8",
        )
        if r:
            results.append(r)

    if "openvino" in EXPORT_FORMATS:
        r = export_one(
            model,
            data_yaml,
            "openvino",
            format="openvino",
            half=True,
            label="OpenVINO FP16",
        )
        if r:
            results.append(r)

    return results


def main():
    check_environment()
    print(f"資料集對應: {DATA_YAML_BY_KEYWORD}，預設: {DATA_YAML_DEFAULT}")
    print(f"導出格式: {EXPORT_FORMATS}")
    print(f"待處理模型數: {len(MODELS_LIST)}\n")

    all_paths = []
    for i, model_path in enumerate(MODELS_LIST):
        print(f"\n--- 進度 {i+1}/{len(MODELS_LIST)} ---")
        all_paths.extend(run_export_for_model(model_path))

    print("\n" + "="*60)
    print("導出完成。可將以下路徑加入 predict.py 的 MODELS_LIST 進行評測：")
    print("-"*60)
    for p in all_paths:
        print(f"    '{p}',")
    print("="*60)


if __name__ == "__main__":
    main()
