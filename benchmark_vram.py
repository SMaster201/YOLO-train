"""
單獨測試每個量化模型佔用的 VRAM。
依序載入每個模型，測量「載入後」、以及「跑測試資料集推理時」的 GPU 記憶體峰值，並輸出表格與 CSV。
"""
import sys
import yaml
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# ================= 設定區 =================
# 測試集資料 YAML（用於取得 test 影像路徑）
DATA_YAML = 'data_mosaic.yaml'
# 測試集最多跑幾張圖（0 = 全部），避免單一模型跑太久
MAX_TEST_IMAGES = 50

# 要測試的模型路徑（與 predict.py 一致，可自行增減）
MODELS_LIST = [
    'runs/detect/768/yolo26_mosaic_nano/weights/best.pt',
    'runs/detect/768/yolo26_mosaic_nano/weights/best_fp16.engine',
    'runs/detect/768/yolo26_mosaic_nano/weights/best_int8.engine',

    'runs/detect/768/yolo26_mosaic_small/weights/best.pt',
    'runs/detect/768/yolo26_mosaic_small/weights/best_fp16.engine',
    'runs/detect/768/yolo26_mosaic_small/weights/best_int8.engine',

    'runs/detect/768/yolo26_mosaic_medium/weights/best.pt',
    'runs/detect/768/yolo26_mosaic_medium/weights/best_fp16.engine',
    'runs/detect/768/yolo26_mosaic_medium/weights/best_int8.engine',

    'runs/detect/768/yolo26_mosaic_large/weights/MOSAIC.pt',
    'runs/detect/768/yolo26_mosaic_large/weights/MOSAIC_fp16.engine',
    'runs/detect/768/yolo26_mosaic_large/weights/MOSAIC_int8.engine',

    'runs/detect/768/yolo26_mosaic_xlarge/weights/MOSAIC.pt',
    'runs/detect/768/yolo26_mosaic_xlarge/weights/MOSAIC_fp16.engine',
    'runs/detect/768/yolo26_mosaic_xlarge/weights/MOSAIC_int8.engine',
]

# 推理時使用的影像尺寸（須與模型訓練/導出時一致）
IMGSZ = 768

# 輸出的 CSV 檔名（可選，留空則不寫檔）
OUTPUT_CSV = 'vram_benchmark.csv'
# ========================================


def get_dataset_paths_from_yaml(yaml_path: str, split: str = 'test') -> tuple[Path | None, Path | None]:
    """解析 data YAML，回傳 (images_dir, labels_dir)。"""
    try:
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        path_root = data.get('path', '.')
        root_path = Path(path_root) if Path(path_root).is_absolute() else yaml_path.parent / path_root
        rel_path = data.get(split, 'test')
        if isinstance(rel_path, list):
            rel_path = rel_path[0]
        images_path = root_path / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        if not images_path.exists():
            return None, None
        if 'images' in str(images_path):
            labels_path = Path(str(images_path).replace('images', 'labels'))
            if labels_path.exists():
                return images_path, labels_path
        labels_path_alt = root_path / 'labels' / rel_path
        if labels_path_alt.exists():
            return images_path, labels_path_alt
        if list(images_path.glob('*.txt')):
            return images_path, images_path
        return images_path, None
    except Exception:
        return None, None


def get_test_image_paths(data_yaml: str, max_images: int = 0) -> list[Path]:
    """從 data YAML 取得 test 集影像路徑列表。max_images=0 表示全部。"""
    exts = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
    images_dir, _ = get_dataset_paths_from_yaml(data_yaml, split='test')
    if not images_dir or not images_dir.exists():
        return []
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def get_quantization_label(path_str: str) -> str:
    """從檔名推斷量化類型。"""
    name = Path(path_str).name.lower()
    if 'int8' in name:
        return 'INT8'
    if 'fp16' in name:
        return 'FP16'
    return 'FP32'


def get_variant_from_path(path_str: str) -> str:
    """從路徑推斷 variant（nano/small/medium/large/xlarge）。"""
    path_str_lower = path_str.lower()
    for v in ('xlarge', 'large', 'medium', 'small', 'nano'):
        if v in path_str_lower:
            return v.capitalize()
    return '-'


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def measure_vram_for_model(
    model_path: str,
    imgsz: int,
    test_image_paths: list[Path] | None = None,
) -> dict | None:
    """
    載入單一模型，測量載入後 VRAM，以及「跑測試資料集推理時」的峰值 VRAM。
    回傳 dict 含 model_name, variant, quantization, vram_loaded_mb, vram_test_peak_mb，失敗回傳 None。
    """
    path = Path(model_path)
    if not path.exists():
        print(f"  [Skip] 檔案不存在: {model_path}")
        return None

    model_name = path.name
    quant = get_quantization_label(model_path)
    variant = get_variant_from_path(model_path)

    if not torch.cuda.is_available():
        print("  [Error] CUDA 不可用，無法測量 VRAM")
        return None

    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    try:
        # 載入前 baseline
        mem_before = torch.cuda.memory_allocated(device)

        model = YOLO(str(path))

        torch.cuda.synchronize()
        vram_after_load = torch.cuda.memory_allocated(device)
        vram_loaded_mb = bytes_to_mb(vram_after_load - mem_before)

        # 跑測試集推理並記錄峰值 VRAM
        torch.cuda.reset_peak_memory_stats()
        if test_image_paths:
            for img_path in test_image_paths:
                _ = model(str(img_path), imgsz=imgsz, verbose=False)
        else:
            dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            _ = model(dummy_img, imgsz=imgsz, verbose=False)
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        vram_test_peak_mb = bytes_to_mb(peak_bytes)

        return {
            'model_name': model_name,
            'variant': variant,
            'quantization': quant,
            'vram_loaded_mb': round(vram_loaded_mb, 2),
            'vram_test_peak_mb': round(vram_test_peak_mb, 2),
        }
    except Exception as e:
        print(f"  [Error] {model_name}: {e}")
        return None
    finally:
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    print("=" * 60)
    print("量化模型 VRAM 佔用測試（含測試集推理）")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("錯誤：未偵測到 CUDA，請在具備 GPU 的環境下執行。")
        sys.exit(1)

    # 取得測試集影像路徑
    test_image_paths = get_test_image_paths(DATA_YAML, MAX_TEST_IMAGES)
    if not test_image_paths:
        print(f"警告：無法從 {DATA_YAML} 取得測試集影像，將改為單張假圖推理測峰值。")
    else:
        print(f"測試集影像數: {len(test_image_paths)}（來自 {DATA_YAML}）")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"影像尺寸: {IMGSZ}x{IMGSZ}")
    print(f"模型數量: {len(MODELS_LIST)}")
    print()

    results = []
    for i, model_path in enumerate(MODELS_LIST, 1):
        print(f"[{i}/{len(MODELS_LIST)}] {Path(model_path).name} ... ", end="", flush=True)
        row = measure_vram_for_model(model_path, IMGSZ, test_image_paths=test_image_paths or None)
        if row:
            results.append(row)
            print(f"載入 {row['vram_loaded_mb']} MB, 測試集推理峰值 {row['vram_test_peak_mb']} MB")
        else:
            print("失敗")

    if not results:
        print("沒有成功測量任何模型。")
        sys.exit(1)

    # 表格輸出
    print()
    print("-" * 75)
    header = f"{'模型名稱':<35} {'Variant':<8} {'精度':<6} {'載入 VRAM (MB)':>14} {'測試集推理峰值 (MB)':>20}"
    print(header)
    print("-" * 75)
    for r in results:
        print(f"{r['model_name']:<35} {r['variant']:<8} {r['quantization']:<6} {r['vram_loaded_mb']:>14.2f} {r['vram_test_peak_mb']:>20.2f}")
    print("-" * 75)

    # 寫入 CSV
    if OUTPUT_CSV:
        import csv
        csv_path = Path(OUTPUT_CSV)
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(
                f,
                fieldnames=['model_name', 'variant', 'quantization', 'vram_loaded_mb', 'vram_test_peak_mb'],
            )
            w.writeheader()
            w.writerows(results)
        print(f"\n已寫入: {csv_path.resolve()}")

    print("\n完成。")


if __name__ == '__main__':
    main()
