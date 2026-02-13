import time
import sys
import csv
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
import wandb
import os

# 嘗試導入 supervision 與 MeanAverageRecall（mAR 在 supervision.metrics 子模組）
try:
    import supervision as sv
    try:
        from supervision.metrics import MeanAverageRecall as _MeanAverageRecall
        _MAR_CLASS = _MeanAverageRecall
    except AttributeError:
        _MAR_CLASS = getattr(sv, "MeanAverageRecall", None)
    SUPERVISION_AVAILABLE = bool(_MAR_CLASS)
    if SUPERVISION_AVAILABLE:
        print(f" [Info] Supervision version: {sv.__version__} (mAR 可用)")
    else:
        print(" [Warning] Supervision 已安裝但無 MeanAverageRecall，請升級: pip install -U supervision")
except ImportError:
    SUPERVISION_AVAILABLE = False
    _MAR_CLASS = None
    print(" [Warning] Supervision library not found. pip install supervision for accurate mAR metrics.")

# ================= 設定區 =================
DATA_YAML = 'data_mosaic.yaml'

# 輸入影像尺寸（須與訓練/導出時一致，例如 640 或 768）
IMGSZ = 768

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
# ========================================

METRICS_HEADERS = [
    '模型名稱', 'Variant', 'Parameters (M)', 'Quantization', 'Precision', 'Recall', 'F1-Score',
    'mAP50', 'mAP50-95', 'mAR50', 'mAR50-95', 
    'Avg Preprocess (s)', 'Avg Inference (s)', 'Avg Postprocess (s)', 'Avg Total (s)', 'Total Render Time (s)',
    'Avg GFLOPS', 'Save Dir'
]

INFO_CACHE = {}

def get_clean_stem(path_str):
    stem = Path(path_str).stem
    clean = stem.replace('_fp16', '').replace('_FP16', '') \
                .replace('_int8', '').replace('_INT8', '') \
                .replace('_fp32', '').replace('_FP32', '') \
                .replace('_quant', '')
    if clean.endswith('_'): clean = clean[:-1]
    return clean

def determine_model_variant(params_count):
    if params_count < 5_000_000: return 'Nano'
    if params_count < 18_000_000: return 'Small'
    if params_count < 24_000_000: return 'Medium'
    if params_count < 55_000_000: return 'Large'
    return 'XLarge'

def pre_scan_models_info(models_list, imgsz=640):
    print("正在預先掃描 FP32 模型資訊以校正 GFLOPS 與參數量...")
    for model_path in models_list:
        path = Path(model_path)
        if path.suffix.lower() == '.pt' and path.exists():
            try:
                model = YOLO(str(path))
                print(f"  [Info] 正在融合模型層 (Fusing): {path.name}")
                try: model.fuse()
                except Exception: pass

                stem = get_clean_stem(path)
                parent_dir = str(path.parent.resolve())
                
                gflops = 0.0
                try:
                    if hasattr(model, 'model'):
                        gflops = get_flops(model.model, imgsz)
                except: pass
                
                params_count = 0
                try:
                    params_count = sum(p.numel() for p in model.model.parameters())
                except: pass
                
                variant = determine_model_variant(params_count)
                
                entry = {
                    'gflops': round(gflops, 4),
                    'variant': variant,
                    'params': round(params_count / 1e6, 2)
                }
                INFO_CACHE[(stem, parent_dir)] = entry
                if stem not in INFO_CACHE:
                    INFO_CACHE[stem] = entry
            except Exception as e:
                print(f"  [Skip] 無法讀取 {path.name}: {e}")
    print("預掃描完成。\n")

def get_dataset_paths_from_yaml(yaml_path, split='test'):
    """解析 YAML 並嘗試推斷 images 和 labels 資料夾路徑（支援兩種結構）"""
    try:
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        path_root = data.get('path', '.')
        root_path = Path(path_root) if Path(path_root).is_absolute() else yaml_path.parent / path_root
        
        rel_path = data.get(split, 'test')
        if isinstance(rel_path, list): rel_path = rel_path[0]
            
        images_path = root_path / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        
        if not images_path.exists():
            return None, None
        
        # 策略 1: 檢查是否為標準 YOLO 結構 (images/train <-> labels/train)
        if 'images' in str(images_path):
            labels_path = Path(str(images_path).replace('images', 'labels'))
            if labels_path.exists():
                return images_path, labels_path
        
        # 策略 2: 檢查是否有獨立的 labels 目錄 (dataset/labels/train)
        labels_path_alt = root_path / 'labels' / rel_path
        if labels_path_alt.exists():
            return images_path, labels_path_alt
        
        # 策略 3: 圖片和標籤在同一目錄（簡化結構，test/ 同時有 .bmp 和 .txt）
        # 檢查該目錄是否有 .txt 標籤文件
        txt_files = list(images_path.glob('*.txt'))
        if txt_files:
            # 同一目錄，返回相同路徑
            return images_path, images_path
        
        # 策略 4: 嘗試平級 labels 目錄
        labels_path_fallback = images_path.parent.parent / 'labels' / images_path.name
        if labels_path_fallback.exists():
            return images_path, labels_path_fallback
        
        # 都找不到，返回 images_path 和 None（讓 Supervision 嘗試從同目錄讀取）
        return images_path, None
        
    except Exception as e:
        print(f" [Warning] YAML 解析路徑失敗: {e}")
        return None, None


def _empty_detections():
    """回傳空的 sv.Detections（相容各版 supervision）。"""
    try:
        return sv.Detections.empty()
    except AttributeError:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            class_id=np.array([], dtype=int)
        )


def _load_yolo_labels_to_detections(label_path, img_w, img_h, encoding='utf-8'):
    """從 YOLO 格式 .txt 讀取標註並轉為 sv.Detections（xyxy, class_id）。"""
    if not Path(label_path).exists():
        return _empty_detections()
    try:
        with open(label_path, 'r', encoding=encoding, errors='replace') as f:
            lines = [L.strip() for L in f if L.strip()]
    except Exception:
        return _empty_detections()
    if not lines:
        return _empty_detections()
    xyxy_list = []
    class_ids = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            clsid = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except (ValueError, IndexError):
            continue
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        xyxy_list.append([x1, y1, x2, y2])
        class_ids.append(clsid)
    if not xyxy_list:
        return _empty_detections()
    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int)
    )


def _load_mar_dataset_manual(images_dir, labels_dir, data_yaml, ext=('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')):
    """手動載入圖片與 YOLO 標籤（全程 UTF-8），回傳 (images, targets) 列表。"""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else images_dir
    try:
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        nc = int(data.get('nc', 0))
    except Exception:
        nc = 0
    try:
        import cv2
    except ImportError:
        from PIL import Image
        def _read_size(p):
            im = Image.open(p)
            return im.size[0], im.size[1]
    else:
        def _read_size(p):
            im = cv2.imread(str(p))
            return (im.shape[1], im.shape[0]) if im is not None else (IMGSZ, IMGSZ)
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in ext:
            continue
        label_path = labels_dir / (img_path.stem + '.txt')
        w, h = _read_size(img_path)
        target = _load_yolo_labels_to_detections(label_path, w, h, encoding='utf-8')
        # 讀取影像供後續推理（路徑傳給 model 即可，這裡只建 target 列表）
        pairs.append((str(img_path), target))
    return pairs


def calculate_mar_supervision(model, data_yaml):
    """使用 Supervision 計算 mAR50 和 mAR50-95（API: supervision.metrics.MeanAverageRecall）"""
    if not SUPERVISION_AVAILABLE or _MAR_CLASS is None:
        return None, None

    print(" [Supervision] 正在啟動 mAR 計算流程 (這可能需要一點時間)...")
    images_dir, labels_dir = get_dataset_paths_from_yaml(data_yaml, split='test')
    
    if not images_dir or not images_dir.exists():
        print(" [Supervision] 找不到影像目錄，跳過 mAR 計算。")
        return None, None
    
    # 使用 supervision.metrics.MeanAverageRecall（內建 IoU 0.5~0.95）
    metric = _MAR_CLASS()

    dataset = None
    use_manual_load = False
    try:
        if labels_dir and labels_dir.exists() and labels_dir == images_dir:
            dataset = sv.DetectionDataset.from_yolo(
                images_directory_path=str(images_dir),
                annotations_directory_path=str(images_dir),
                data_yaml_path=str(data_yaml)
            )
        elif labels_dir and labels_dir.exists():
            dataset = sv.DetectionDataset.from_yolo(
                images_directory_path=str(images_dir),
                annotations_directory_path=str(labels_dir),
                data_yaml_path=str(data_yaml)
            )
        else:
            dataset = sv.DetectionDataset.from_yolo(
                images_directory_path=str(images_dir),
                annotations_directory_path=str(images_dir),
                data_yaml_path=str(data_yaml)
            )
    except (UnicodeDecodeError, UnicodeError) as e:
        print(f" [Supervision] 編碼錯誤 (cp950)，改用手動載入 (UTF-8): {e}")
        use_manual_load = True
    except Exception as e:
        print(f" [Supervision] 數據集載入失敗: {e}")
        return None, None

    if use_manual_load:
        dataset = None

    predictions_list = []
    targets_list = []

    if use_manual_load or dataset is None:
        try:
            pairs = _load_mar_dataset_manual(images_dir, labels_dir or images_dir, data_yaml)
        except Exception as e:
            print(f" [Supervision] 手動載入數據集失敗: {e}")
            return None, None
        if not pairs:
            print(" [Supervision] 手動載入後無有效樣本。")
            return None, None
        print(f" [Supervision] 正在對 {len(pairs)} 張影像進行推理 (手動載入)...")
        for img_path, target in pairs:
            results = model(img_path, imgsz=IMGSZ, verbose=False)
            result = results[0]
            detections = sv.Detections.from_ultralytics(result)
            predictions_list.append(detections)
            targets_list.append(target)
    else:
        print(f" [Supervision] 正在對 {len(dataset)} 張影像進行推理...")
        for img_name, image, target in dataset:
            results = model(image, imgsz=IMGSZ, verbose=False)
            result = results[0]
            detections = sv.Detections.from_ultralytics(result)
            predictions_list.append(detections)
            targets_list.append(target)

    print(" [Supervision] 正在計算指標...")
    metric.update(predictions_list, targets_list)
    res = metric.compute()

    # recall_per_class: (num_classes, num_iou_thresholds), iou_thresholds 通常為 0.5, 0.55, ..., 0.95
    recall_per_class = getattr(res, "recall_per_class", None)
    if recall_per_class is None or recall_per_class.size == 0:
        # fallback: 新 API 的 recall_scores 為 mAR@1, mAR@10, mAR@100，取 mAR@100 當近似
        rs = getattr(res, "recall_scores", None)
        if rs is not None and len(rs) > 0:
            val_mar50_95 = val_mar50 = float(np.mean(rs))
            print(f" [Supervision] mAR@50: {val_mar50:.4f}, mAR@50-95: {val_mar50_95:.4f} (fallback)")
            return val_mar50, val_mar50_95
        return None, None
    recall_per_class = np.asarray(recall_per_class)
    # mAR@50 = 平均 recall at IoU 0.5 (第一欄)
    val_mar50 = float(np.mean(recall_per_class[:, 0]))
    # mAR@50-95 = 全閾值平均
    val_mar50_95 = float(np.mean(recall_per_class))

    print(f" [Supervision] mAR@50: {val_mar50:.4f}, mAR@50-95: {val_mar50_95:.4f}")
    return val_mar50, val_mar50_95

def evaluate_single_model(model_path, data_yaml='data.yaml'):
    model_path = str(model_path)
    model_name = Path(model_path).name
    clean_stem = get_clean_stem(model_name)
    print(f"\n[{model_name}] 正在評估模型 ...")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"無法載入模型: {e}"); return None

    # 1. 執行 Ultralytics 標準驗證 (獲取 mAP, Speed, Confusion Matrix)
    try:
        result = model.val(data=data_yaml, imgsz=IMGSZ, verbose=False, save_json=True)
        save_dir = getattr(result, 'save_dir', None)
        metrics = getattr(result, 'metrics', result)
        box = metrics.box
        
        precision = float(box.mp)
        recall = float(box.mr)
        map50 = float(box.map50)
        map50_95 = float(box.map)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 混淆矩陣
        total_tp = total_fp = total_fn = 0
        per_class = []
        cm = getattr(result, 'confusion_matrix', None)
        if cm and hasattr(cm, 'matrix'):
            mat = np.array(cm.matrix)
            nc = mat.shape[0] - 1
            if nc > 0:
                total_tp = int(np.diag(mat[:nc, :nc]).sum())
                total_fp = int(mat[:nc, nc].sum()) 
                total_fn = int(mat[nc, :nc].sum())
                nt_per_class = getattr(metrics, 'nt_per_class', [0]*nc)
                for c in range(nc):
                    gt = int(nt_per_class[c])
                    matched = int(mat[c, c])
                    pred_c = matched + int(mat[:nc, c].sum()) - matched
                    per_class.append({'class_id': c, 'name': result.names.get(c, str(c)), 'gt': gt, 'matched': matched, 'precision': matched/pred_c if pred_c>0 else 0, 'recall': matched/gt if gt>0 else 0})
        
        avg_count_accuracy = max(0.0, 1.0 - (abs((total_tp+total_fp) - (total_tp+total_fn)) / (total_tp+total_fn))) if (total_tp+total_fn) > 0 else 0.0

    except Exception as e:
        print(f"驗證計算失敗: {e}"); return None

    # 2. 計算 mAR (使用 Supervision)
    mar50, mar50_95 = None, None
    if SUPERVISION_AVAILABLE:
        try:
            mar50, mar50_95 = calculate_mar_supervision(model, data_yaml)
        except Exception as e:
            print(f" [Error] Supervision mAR 計算失敗: {e}")

    # 如果 Supervision 失敗或沒安裝，使用 fallback (注意：Ultralytics 預設不提供 mAR50-95)
    if mar50 is None: mar50 = recall # Fallback to Recall@0.5
    if mar50_95 is None: mar50_95 = 0.0 # Fallback to 0 to indicate missing data

    # 3. 時間與輔助資訊
    speed = getattr(result, 'speed', {})
    avg_pre_sec, avg_inf_sec, avg_post_sec = speed.get('preprocess', 0)/1000, speed.get('inference', 0)/1000, speed.get('postprocess', 0)/1000
    avg_total_sec = avg_pre_sec + avg_inf_sec + avg_post_sec
    total_images = getattr(metrics, 'n_img', 0) or 0
    if total_images == 0:
        img_dir, _ = get_dataset_paths_from_yaml(data_yaml, split='test')
        if img_dir and img_dir.exists():
            exts = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
            total_images = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in exts)

    # 處理 GFLOPS / Params Cache
    parent_dir = str(Path(model_path).parent.resolve())
    cached = INFO_CACHE.get((clean_stem, parent_dir), INFO_CACHE.get(clean_stem))
    gflops = cached['gflops'] if cached else 0.0
    params_m = cached['params'] if cached else 0.0
    variant = cached['variant'] if cached else 'unknown'

    return {
        '模型名稱': model_name, 'Variant': variant, 'Parameters (M)': params_m, 
        'Quantization': 'INT8' if 'int8' in model_name.lower() else ('FP16' if 'fp16' in model_name.lower() else 'FP32'),
        'Precision': precision, 'Recall': recall, 'F1-Score': f1,
        'mAP50': map50, 'mAP50-95': map50_95, 
        'mAR50': mar50, 'mAR50-95': mar50_95, 
        'Avg Preprocess (s)': avg_pre_sec, 'Avg Inference (s)': avg_inf_sec, 'Avg Postprocess (s)': avg_post_sec, 
        'Avg Total (s)': avg_total_sec, 'Total Render Time (s)': avg_total_sec * total_images,
        'Avg GFLOPS': gflops, 'save_dir': save_dir, 'total_images': total_images,
        'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn, 
        'per_class': per_class, 'avg_count_accuracy': avg_count_accuracy
    }

def save_summary_txt(row):
    output_dir = Path('result')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clean_stem = get_clean_stem(row['模型名稱'])
    quant = row['Quantization']
    variant = row['Variant']
    
    filename = f"{clean_stem}_{variant}_{quant}_{timestamp}_summary.txt"
    filepath = output_dir / filename
    
    fmt = lambda v, d=3: f"{v:.{d}f}"
    mar95_str = fmt(row['mAR50-95']) if row['mAR50-95'] > 0 else "N/A"

    lines = [
        f"Evaluation Summary for: {row['模型名稱']}",
        "============================================================",
        f"Total images: {row['total_images']}",
        f"mAP@50: {fmt(row['mAP50'])} | mAP@50-95: {fmt(row['mAP50-95'])}",
        f"mAR@50: {fmt(row['mAR50'])} | mAR@50-95: {mar95_str}",
        f"Precision: {fmt(row['Precision'])} | Recall: {fmt(row['Recall'])} | F1: {fmt(row['F1-Score'])}",
        "",
        "Speed / Latency Breakdown (ms):",
        f"  Preprocess:  {fmt(row['Avg Preprocess (s)']*1000, 2)} ms",
        f"  Inference:   {fmt(row['Avg Inference (s)']*1000, 2)} ms", 
        f"  Postprocess: {fmt(row['Avg Postprocess (s)']*1000, 2)} ms",
        f"  Total Avg:   {fmt(row['Avg Total (s)']*1000, 2)} ms",
        f"Dataset Total Time: {fmt(row['Total Render Time (s)'], 2)} s",
        "",
        "Model Details:",
        f"  Variant: {row['Variant']} | Params: {row['Parameters (M)']} M | GFLOPS: {row['Avg GFLOPS']}",
        "============================================================"
    ]
    
    filepath.write_text('\n'.join(lines), encoding='utf-8')
    print(f" [v] 本地文檔已儲存: {filepath.name}")

def save_all_metrics_csv(rows, filepath='all_models_benchmark.csv'):
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(METRICS_HEADERS)
        for r in rows:
            w.writerow([
                r['模型名稱'], r['Variant'], r['Parameters (M)'], r['Quantization'], 
                f"{r['Precision']:.6f}", f"{r['Recall']:.6f}", f"{r['F1-Score']:.6f}",
                f"{r['mAP50']:.6f}", f"{r['mAP50-95']:.6f}",
                f"{r['mAR50']:.6f}", f"{r['mAR50-95']:.6f}", 
                f"{r['Avg Preprocess (s)']:.6f}", f"{r['Avg Inference (s)']:.6f}", 
                f"{r['Avg Postprocess (s)']:.6f}", f"{r['Avg Total (s)']:.6f}", f"{r['Total Render Time (s)']:.6f}",
                f"{r['Avg GFLOPS']:.4f}", str(r['save_dir'])
            ])

if __name__ == '__main__':
    wandb.init(project="yolo26-eval", name=time.strftime("%Y%m%d_%H%M%S"))
    pre_scan_models_info(MODELS_LIST, imgsz=IMGSZ)
    all_results = []
    for model_path in MODELS_LIST:
        if not Path(model_path).exists(): continue
        row = evaluate_single_model(model_path, data_yaml=DATA_YAML)
        if row:
            save_summary_txt(row)
            all_results.append(row)
            wandb.log({
                "model/name": row["模型名稱"],
                "model/variant": row["Variant"],
                "model/params_M": row["Parameters (M)"],
                "model/gflops": row["Avg GFLOPS"],
                "metrics/mAP50": row["mAP50"],
                "metrics/mAP50-95": row["mAP50-95"],
                "metrics/mAR50": row["mAR50"],
                "metrics/mAR50-95": row["mAR50-95"],
                "metrics/f1_score": row["F1-Score"],
                "latency/preprocess_ms": row["Avg Preprocess (s)"] * 1000,
                "latency/inference_ms": row["Avg Inference (s)"] * 1000,
                "latency/postprocess_ms": row["Avg Postprocess (s)"] * 1000,
                "latency/total_avg_ms": row["Avg Total (s)"] * 1000,
                "latency/total_render_time_s": row["Total Render Time (s)"], 
            })
    if all_results: save_all_metrics_csv(all_results)
    wandb.finish()