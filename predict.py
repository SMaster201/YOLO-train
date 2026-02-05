import time
import sys
import csv
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
import wandb

# ================= 設定區 =================
# 1. 請在這裡設定你的資料設定檔 (data.yaml) 的路徑
DATA_YAML = 'data_mnb.yaml'

# 2. 請將你想測試的所有模型路徑放入此列表 
# (建議包含 .pt 作為基準，這樣 .engine 才能繼承參數量資訊)
MODELS_LIST = [
    # 範例路徑，請修改為你實際的路徑
    'runs/detect/yolo26_project5/weights/MNB.pt',           # FP32 基準
    'runs/detect/yolo26_project5/weights/MNB_fp16.engine',  # TensorRT FP16
    'runs/detect/yolo26_project5/weights/MNB_int8.engine',  # TensorRT INT8

    'runs/detect/yolo26_Small/weights/MNB.pt',           # FP32 基準
    'runs/detect/yolo26_Small/weights/MNB_fp16.engine',  # TensorRT FP16
    'runs/detect/yolo26_Small/weights/MNB_int8.engine',  # TensorRT INT8

    'runs/detect/yolo26_mnb_medium/weights/MNB.pt',           # FP32 基準
    'runs/detect/yolo26_mnb_medium/weights/MNB_fp16.engine',  # TensorRT FP16
    'runs/detect/yolo26_mnb_medium/weights/MNB_int8.engine'  # TensorRT INT8
]
# ========================================

METRICS_HEADERS = [
    '模型名稱', 'Variant', 'Parameters (M)', 'Quantization', 'Precision', 'Recall', 'F1-Score',
    'Avg Preprocess (s)', 'Avg Inference (s)', 'Avg Postprocess (s)', 'Avg Total (s)',
    'Avg GFLOPS', 'Save Dir'
]

# 用於儲存 FP32 模型的資訊 (GFLOPS, Variant, Parameters)，供量化模型查詢使用
# Key: (stem, parent_dir) 避免不同目錄的同名 .pt 互相覆蓋；查詢時可 fallback 到單一 stem
INFO_CACHE = {}

def get_clean_stem(path_str):
    """移除量化後綴，取得原始模型名稱 (例如 MNB_int8 -> MNB)"""
    stem = Path(path_str).stem
    clean = stem.replace('_fp16', '').replace('_FP16', '') \
                .replace('_int8', '').replace('_INT8', '') \
                .replace('_fp32', '').replace('_FP32', '') \
                .replace('_quant', '')
    if clean.endswith('_'): clean = clean[:-1]
    return clean

def determine_model_variant(params_count):
    """根據參數量判斷模型大小 (Nano/Small/Medium/Large/XLarge)"""
    # 這裡傳入的是實際數量 (例如 3000000)
    if params_count < 5_000_000: return 'Nano'
    if params_count < 18_000_000: return 'Small'
    if params_count < 35_000_000: return 'Medium'
    if params_count < 55_000_000: return 'Large'
    return 'XLarge'

def pre_scan_models_info(models_list, imgsz=640):
    """
    預先掃描所有 .pt 檔案，執行 Fuse，計算 GFLOPS、參數量和模型大小並快取。
    """
    print("正在預先掃描 FP32 模型資訊以校正 GFLOPS 與參數量...")
    for model_path in models_list:
        path = Path(model_path)
        # 只掃描 .pt 檔，因為 .engine 無法讀取結構
        if path.suffix.lower() == '.pt' and path.exists():
            try:
                model = YOLO(str(path))
                
                # ============================================
                # [關鍵修正] 強制執行層融合 (Fuse Conv+BN)
                # 這會讓參數統計符合「推論狀態」，數值會比訓練時略少
                # ============================================
                print(f"  [Info] 正在融合模型層 (Fusing) 以取得推論參數: {path.name}")
                try:
                    model.fuse()
                except Exception as fe:
                    print(f"  [Warn] Fuse 失敗 (可能是已經融合過或不支援): {fe}")

                stem = get_clean_stem(path)
                parent_dir = str(path.parent.resolve())
                
                # 1. 計算 GFLOPS (基於融合後的模型)
                gflops = 0.0
                try:
                    if hasattr(model, 'model'):
                        gflops = get_flops(model.model, imgsz)
                except: pass
                
                # 2. 計算參數量 (Parameters)
                params_count = 0
                try:
                    params_count = sum(p.numel() for p in model.model.parameters())
                except: pass
                
                # 3. 判斷 Variant
                variant = determine_model_variant(params_count)
                
                # 存入快取：用 (stem, parent_dir) 當 key，避免不同目錄的同名 FP32 互相覆蓋
                entry = {
                    'gflops': round(gflops, 4),
                    'variant': variant,
                    'params': round(params_count / 1e6, 2) # Store as Millions
                }
                INFO_CACHE[(stem, parent_dir)] = entry
                # 若此 stem 尚未有「僅 stem」的 fallback，也存一份（僅在第一次出現時）
                if stem not in INFO_CACHE:
                    INFO_CACHE[stem] = entry
                # print(f"  -> {stem}: {variant}, {entry['params']}M params (Fused)")
            except Exception as e:
                print(f"  [Skip] 無法讀取 {path.name}: {e}")
    print("預掃描完成。\n")

def get_test_path_from_yaml(yaml_path):
    """從 data.yaml 取得測試集路徑"""
    try:
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        root = data.get('path', '.')
        root_path = Path(root) if Path(root).is_absolute() else yaml_path.parent / root
        test_rel = data.get('test', 'test')
        test_path = root_path / (test_rel if isinstance(test_rel, str) else 'test')
        if test_path.exists() and test_path.is_dir():
            return str(test_path.resolve())
        return None
    except Exception:
        return None

def get_image_count_from_yaml(yaml_path, split='test'):
    """從 data.yaml 讀取測試集路徑並強制計算圖片數量"""
    count = 0
    try:
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        path_str = data.get(split)
        if not path_str and split == 'test':
            path_str = 'test'
        if not path_str:
            return 0
        root = data.get('path', '.')
        root_path = Path(root) if Path(root).is_absolute() else yaml_path.parent / root
        dataset_path = root_path / path_str if not Path(path_str).is_absolute() else Path(path_str)
        if not dataset_path.exists():
            dataset_path = yaml_path.parent / path_str

        if dataset_path.exists() and dataset_path.is_dir():
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
            count = len([x for x in dataset_path.rglob('*') if x.suffix.lower() in exts])
        elif dataset_path.suffix == '.txt':
             with open(dataset_path, 'r') as f:
                 count = len(f.readlines())
    except Exception as e:
        print(f"警告：計算圖片數量時發生錯誤 ({e})")
    return count if count > 0 else 0

def get_quantization_label(model_path):
    path = str(Path(model_path).name).lower()
    if 'int8' in path or 'quantized' in path or 'tflite' in path: return 'INT8'
    if 'half' in path or 'fp16' in path: return 'FP16'
    return 'FP32'

def evaluate_single_model(model_path, data_yaml='data.yaml', warmup=5, repeat=20):
    """執行評估單一模型並計算所有數值"""
    model_path = str(model_path)
    model_name = Path(model_path).name
    clean_stem = get_clean_stem(model_name)
    print(f"\n[{model_name}] 正在評估模型 ...")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"無法載入模型: {e}")
        return None

    # 1. 取得測試集圖片總數
    total_images = get_image_count_from_yaml(data_yaml, split='test')
    if total_images == 0:
        print("警告: 找不到測試集圖片，數據可能不準確。")

    # 2. 執行驗證 (model.val)
    save_dir = None
    try:
        result = model.val(data=data_yaml, verbose=False)
        if hasattr(result, 'save_dir'):
            save_dir = result.save_dir
        
        metrics = getattr(result, 'metrics', result)
        box = metrics.box
        
        precision = float(box.mp) if hasattr(box, 'mp') else 0.0
        recall = float(box.mr) if hasattr(box, 'mr') else 0.0
        map50_95 = float(box.map) if hasattr(box, 'map') else 0.0 
        
        if hasattr(box, 'f1') and getattr(box, 'f1', None) is not None and len(box.f1) > 0:
            f1 = float(np.mean(box.f1))
        elif (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        total_tp = 0
        total_fp = 0
        total_fn = 0
        per_class = []
        
        cm = getattr(result, 'confusion_matrix', None)
        names = result.names
        
        if cm and hasattr(cm, 'matrix'):
            mat = np.array(cm.matrix)
            nc = mat.shape[0] - 1
            if nc > 0:
                total_tp = int(np.diag(mat[:nc, :nc]).sum())
                total_fp = int(mat[:nc, nc].sum()) 
                total_fn = int(mat[nc, :nc].sum())
            
            nt_per_class = getattr(metrics, 'nt_per_class', None)
            for c in range(nc):
                gt = int(nt_per_class[c]) if (nt_per_class is not None and c < len(nt_per_class)) else 0
                matched = int(mat[c, c])
                fp_c = int(mat[:, c].sum()) - matched
                pred_c = matched + fp_c
                p_c = matched / pred_c if pred_c > 0 else 0.0
                r_c = matched / gt if gt > 0 else 0.0
                name = names.get(c, str(c))
                per_class.append({
                    'class_id': c, 'name': name, 'gt': gt, 'pred': pred_c,
                    'matched': matched, 'precision': p_c, 'recall': r_c
                })

        total_predictions = total_tp + total_fp
        avg_pred_objects = round(total_predictions / total_images, 2) if total_images > 0 else 0.0
        total_gt = total_tp + total_fn
        if total_gt > 0:
            count_error = abs(total_predictions - total_gt) / total_gt
            avg_count_accuracy = max(0.0, 1.0 - count_error)
        else:
            avg_count_accuracy = 0.0
        avg_iou_accuracy = map50_95

    except Exception as e:
        print(f"驗證計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 3. 時間計算
    speed = getattr(result, 'speed', {})
    inference_ms = speed.get('inference', 0.0)
    preprocess_ms = speed.get('preprocess', 0.0)
    postprocess_ms = speed.get('postprocess', 0.0)
    avg_inf_sec = inference_ms / 1000.0
    avg_pre_sec = preprocess_ms / 1000.0
    avg_post_sec = postprocess_ms / 1000.0
    avg_total_sec = avg_pre_sec + avg_inf_sec + avg_post_sec
    
    # 若 Val 沒有回傳時間，則手動測量 (通常 engine 需要這個)
    if avg_inf_sec < 0.00001:
        try:
            source = get_test_path_from_yaml(data_yaml)
            if not source:
                with open(data_yaml, 'r') as f:
                    d = yaml.safe_load(f)
                    root = Path(d.get('path', '.'))
                    source = str(root / 'test')
            
            if source and Path(source).exists():
                print("  正在手動測量推論速度 (End-to-End)...")
                # Warmup
                for _ in range(warmup):
                    list(model.predict(source=source, verbose=False, max_det=1))
                # Benchmark
                t0 = time.perf_counter()
                for _ in range(repeat):
                    list(model.predict(source=source, verbose=False))
                total_duration = time.perf_counter() - t0
                avg_total_sec = total_duration / repeat
                avg_inf_sec = avg_total_sec 
                avg_pre_sec = 0.0
                avg_post_sec = 0.0
        except Exception:
            pass

    # 4. GFLOPS, Params 與 Variant 處理（先依「同目錄」對應的 FP32 查詢，避免被其他 FP32 覆蓋）
    gflops = 0.0
    params_m = 0.0
    variant = 'unknown'
    parent_dir = str(Path(model_path).parent.resolve())
    cache_key_same_dir = (clean_stem, parent_dir)
    
    def get_cached_info():
        if cache_key_same_dir in INFO_CACHE:
            return INFO_CACHE[cache_key_same_dir]
        if clean_stem in INFO_CACHE:
            return INFO_CACHE[clean_stem]
        return None
    
    cached = get_cached_info()
    
    if Path(model_path).suffix.lower() == '.pt':
        # 如果是 .pt，優先使用同目錄的 Cache（Fuse 過的）；沒有再用現場計算
        if cached:
            gflops = cached['gflops']
            variant = cached['variant']
            params_m = cached['params']
        else:
            try:
                if hasattr(model, 'model'):
                    gflops = get_flops(model.model, 640) / 1e9
                    p_count = sum(p.numel() for p in model.model.parameters())
                    params_m = round(p_count / 1e6, 2)
                    variant = determine_model_variant(p_count)
            except: pass
    else:
        # engine/int8：只從 cache 找（同目錄 FP32 優先）
        if cached:
            gflops = cached['gflops']
            variant = cached['variant']
            params_m = cached['params']
        else:
            variant = 'unknown'
            gflops = 0.0
            params_m = 0.0

    return {
        '模型名稱': model_name,
        'Variant': variant, 
        'Parameters (M)': params_m, 
        'Quantization': get_quantization_label(model_path),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Avg Preprocess (s)': avg_pre_sec,
        'Avg Inference (s)': avg_inf_sec,
        'Avg Postprocess (s)': avg_post_sec,
        'Avg Total (s)': avg_total_sec,
        'Avg GFLOPS': gflops,
        'save_dir': save_dir,
        'total_images': total_images,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'per_class': per_class,
        'avg_pred_objects': avg_pred_objects,
        'avg_iou_accuracy': avg_iou_accuracy,
        'avg_count_accuracy': avg_count_accuracy
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
    
    total_images = row['total_images']
    
    def fmt(v, decimals=3, percent=False):
        if v is None: return '0.000'
        if percent: return f'{v*100:.1f}%'
        return f'{v:.{decimals}f}'

    lines = [
        f'Evaluation Summary for: {row["模型名稱"]}',
        '============================================================',
        f'Total images: {total_images}',
        f'Total TP: {row["total_tp"]}, FP: {row["total_fp"]}, FN: {row["total_fn"]}',
        f'Precision: {fmt(row["Precision"])}',
        f'Recall: {fmt(row["Recall"])}',
        f'F1-Score: {fmt(row["F1-Score"])}',
        '',
        'Model Details:',
        f'  Variant: {row["Variant"]}',
        f'  Parameters (Inference): {fmt(row["Parameters (M)"], 2)} M (Fused)',
        f'  Quantization: {row["Quantization"]}',
        f'  Avg estimated performance: {fmt(row["Avg GFLOPS"], 2)} GFLOPS',
        '',
        'Speed / Latency Breakdown (per image):',
        f'  Preprocess:  {fmt(row["Avg Preprocess (s)"]*1000, 2)} ms',
        f'  Inference:   {fmt(row["Avg Inference (s)"]*1000, 2)} ms',
        f'  Postprocess: {fmt(row["Avg Postprocess (s)"]*1000, 2)} ms',
        f'  Total Time:  {fmt(row["Avg Total (s)"]*1000, 2)} ms',
        '',
        '[Section 1] IoU-Based Matching Statistics:',
        f'  Avg IoU Accuracy: {fmt(row["avg_iou_accuracy"], 3)} (based on mAP@50-95)',
        '',
        '[Section 2] Count-Based Statistics:',
        f'  Avg Count Accuracy: {fmt(row["avg_count_accuracy"], 1, percent=True)}',
        '',
        'Per-Class Summary:',
    ]
    
    for pc in row['per_class']:
        lines.append(
            f'  Class {pc["class_id"]} ({pc["name"]}): '
            f'GT={pc["gt"]} Pred={pc["pred"]} Matched={pc["matched"]} '
            f'P={pc["precision"]:.3f} R={pc["recall"]:.3f}'
        )

    try:
        filepath.write_text('\n'.join(lines), encoding='utf-8')
        print(f"  [v] 已儲存 TXT 報告至: {filepath}")
    except Exception as e:
        print(f"  [x] 儲存 TXT 失敗: {e}")

def save_all_metrics_csv(rows, filepath='all_models_benchmark.csv'):
    filepath = Path(filepath)
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(METRICS_HEADERS)
        for r in rows:
            save_dir_str = str(r['save_dir']) if r['save_dir'] else "N/A"
            w.writerow([
                r['模型名稱'], r['Variant'], r['Parameters (M)'], r['Quantization'], 
                f"{r['Precision']:.6f}", f"{r['Recall']:.6f}", f"{r['F1-Score']:.6f}",
                f"{r['Avg Preprocess (s)']:.6f}", f"{r['Avg Inference (s)']:.6f}", 
                f"{r['Avg Postprocess (s)']:.6f}", f"{r['Avg Total (s)']:.6f}",
                f"{r['Avg GFLOPS']:.4f}", save_dir_str
            ])
    print(f"\n已儲存總表 CSV: {filepath.absolute()}")

def print_metrics_table(rows):
    headers = ['Model', 'Var', 'Params(M)', 'Quant', 'Prec', 'Recall', 'F1', 'Total(ms)', 'SaveDir']
    print("\n" + "=" * 125)
    print(f"{headers[0]:<20} | {headers[1]:<6} | {headers[2]:<9} | {headers[3]:<6} | {headers[4]:<7} | {headers[5]:<7} | {headers[6]:<7} | {headers[7]:<9} | {headers[8]}")
    print("-" * 125)
    for r in rows:
        sd = str(r['save_dir'].name) if r['save_dir'] else "N/A"
        total_ms = r['Avg Total (s)'] * 1000
        print(f"{r['模型名稱']:<20} | {r['Variant']:<6} | {r['Parameters (M)']:<9.2f} | {r['Quantization']:<6} | {r['Precision']:.4f}  | {r['Recall']:.4f}  | {r['F1-Score']:.4f}  | {total_ms:.2f} ms   | {sd}")
    print("=" * 125)

if __name__ == '__main__':
    # 程式／環境資訊（上傳到 W&B）
    try:
        import ultralytics
        ultralytics_version = getattr(ultralytics, "__version__", "?")
    except Exception:
        ultralytics_version = "?"
    program_info = {
        "data_yaml": DATA_YAML,
        "models": MODELS_LIST,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "ultralytics_version": ultralytics_version,
        "script": str(Path(__file__).resolve().name),
        "script_dir": str(Path(__file__).resolve().parent),
    }

    # 初始化 Weights & Biases 專案
    wandb.init(
        project="yolo26-eval",  # 可以改成你自己的專案名稱
        name=time.strftime("%Y%m%d_%H%M%S"),
        config=program_info,
    )

    # Step 0: 預先掃描 FP32 模型以建立 Info Cache (含 Fuse)
    pre_scan_models_info(MODELS_LIST)

    all_results = []
    print(f"準備測試 {len(MODELS_LIST)} 個模型...")
    
    for model_path in MODELS_LIST:
        if not Path(model_path).exists():
            print(f"錯誤: 找不到檔案 {model_path}，跳過。")
            continue
            
        row = evaluate_single_model(model_path, data_yaml=DATA_YAML)
        
        if row:
            save_summary_txt(row)
            all_results.append(row)

            # 上傳到 W&B：Variant、模型名稱、參數量、計算量、渲染時間
            wandb.log({
                "model_name": row["模型名稱"],
                "variant": row["Variant"],
                "params_M": row["Parameters (M)"],
                "gflops": row["Avg GFLOPS"],
                "render_time_s": row["Avg Total (s)"],
            })
        else:
            print(f"模型 {model_path} 評估失敗。")

    if all_results:
        save_all_metrics_csv(all_results)
        print_metrics_table(all_results)
    else:
        print("沒有產生任何有效結果。")

    wandb.finish()