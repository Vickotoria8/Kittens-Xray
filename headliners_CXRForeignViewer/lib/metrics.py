import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
    """Calculate precision and recall for one image"""
    if len(pred_boxes) == 0:
        return 0, 0

    if len(true_boxes) == 0:
        return 0, 1

    tp = 0
    for pred_box in pred_boxes:
        for true_box in true_boxes:
            if calculate_iou(pred_box, true_box) >= iou_threshold:
                tp += 1
                break

    precision = tp / len(pred_boxes)
    recall = tp / len(true_boxes)

    return precision, recall

def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-8)


def calculate_ap(precisions, recalls):
    """Calculate Average Precision from precision-recall curve"""
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    return ap

def calculate_map(predictions, targets, iou_threshold=0.5):
    """Calculate mean Average Precision"""
    aps = []

    for class_id in set([t['label'] for t in targets]):
        precisions = []
        recalls = []
        pred_boxes = []
        true_boxes = []
        for p, t in zip(predictions, targets):
            pred_boxes.append(p['bbox'])
            true_boxes.append(t['bbox'])

            prec, rec = calculate_precision_recall(pred_boxes, true_boxes, iou_threshold)
            precisions.append(prec)
            recalls.append(rec)

        ap = calculate_ap(np.array(precisions), np.array(recalls))
        aps.append(ap)

    return np.mean(aps)

def detection_confusion_matrix(predictions, targets, num_classes, iou_threshold=0.5):
    """Confusion matrix for object detection"""
    cm = np.zeros((num_classes, num_classes))

    for pred, target in zip(predictions, targets):
          pred_label = pred['label']
          pred_box = pred['bbox']

          best_iou = 0
          best_true_label = -1

          true_label = target['label']
          true_box = target['bbox']

          iou = calculate_iou(pred_box, true_box)
          if iou > best_iou and iou >= iou_threshold:
              best_iou = iou
              best_true_label = true_label

          if best_true_label != -1:
                cm[best_true_label, pred_label] += 1
          else:
              # False positive
              cm[:, pred_label] += 1 / num_classes

    return cm

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]:.1f}',
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(scores, labels, roc_auc, iou_threshold):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve
    
    if len(scores) == 0:
        print("No data to plot ROC curve")
        return
        
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title(f'ROC Curve for Object Detection (IoU threshold = {iou_threshold})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Add some important threshold points
    important_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for thresh in important_thresholds:
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - thresh))
        if idx < len(fpr) and idx < len(tpr):
            plt.scatter(fpr[idx], tpr[idx], color='red', s=50, zorder=5)
            plt.annotate(f' {thresh}', (fpr[idx], tpr[idx]), fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print some key metrics
    print(f"Key ROC metrics (IoU thr={iou_threshold}):")
    print(f"  AUC: {roc_auc:.4f}")
    
    # Find threshold where TPR is 0.9
    if len(tpr) > 0:
        idx_90 = np.argmax(tpr >= 0.9)
        if idx_90 < len(fpr) and idx_90 < len(thresholds):
            print(f"  At TPR=0.90: FPR={fpr[idx_90]:.4f}, Threshold={thresholds[idx_90]:.4f}")
        
        # Find threshold where FPR is 0.1
        idx_10 = np.argmax(fpr >= 0.1)
        if idx_10 < len(tpr) and idx_10 < len(thresholds):
            print(f"  At FPR=0.10: TPR={tpr[idx_10]:.4f}, Threshold={thresholds[idx_10]:.4f}")

def plot_comparative_roc_curves(results):
    """Plot all ROC curves on one plot for comparison"""
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    for i, (iou_thresh, result) in enumerate(results.items()):
        if len(result['scores']) > 0:
            fpr, tpr, _ = roc_curve(result['labels'], result['scores'])
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'IoU={iou_thresh} (AUC = {result["roc_auc"]:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title('Comparative ROC Curves for Different IoU Thresholds')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    Format: [x, y, width, height] (YOLO format)
    """
    # Convert from center coordinates to corner coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate corners
    x1_min, x1_max = x1 - w1/2, x1 + w1/2
    y1_min, y1_max = y1 - h1/2, y1 + h1/2
    x2_min, x2_max = x2 - w2/2, x2 + w2/2
    y2_min, y2_max = y2 - h2/2, y2 + h2/2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)  # Исправлено: было inter_x_max
    inter_y_max = min(y1_max, y2_max)  # Исправлено: было inter_y_max
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union

def load_labels_from_folder(folder_path, has_confidence=False):
    """
    Load labels from folder with txt files (YOLO format)
    
    Parameters:
    - folder_path: path to folder with txt files
    - has_confidence: if True, expects format [class, x, y, w, h, confidence]
                     if False, expects format [class, x, y, w, h]
    
    Returns:
    - labels_dict: dictionary {filename: [[x, y, w, h, confidence?], ...]}
    """
    labels_dict = {}
    
    # Find all txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        print(f"Warning: No txt files found in {folder_path}")
        return labels_dict
    
    print(f"Found {len(txt_files)} txt files in {folder_path}")
    
    for txt_file in txt_files:
        filename = os.path.splitext(os.path.basename(txt_file))[0]
        boxes = []
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:  # skip empty lines
                    continue
                    
                parts = line.split()
                
                if has_confidence:
                    # Format for predictions: [class, x, y, w, h, confidence] or [x, y, w, h, confidence]
                    if len(parts) == 6:
                        # [class, x, y, w, h, confidence]
                        x, y, w, h, confidence = map(float, parts[1:6])
                        boxes.append([x, y, w, h, confidence])
                    elif len(parts) == 5:
                        # [x, y, w, h, confidence] - no class
                        x, y, w, h, confidence = map(float, parts[0:5])
                        boxes.append([x, y, w, h, confidence])
                    else:
                        print(f"Warning: Line {line_num} in {txt_file} has {len(parts)} parts, expected 5 or 6")
                        
                else:
                    # Format for ground truth: [class, x, y, w, h] or [x, y, w, h]
                    if len(parts) == 5:
                        # [class, x, y, w, h]
                        x, y, w, h = map(float, parts[1:5])
                        boxes.append([x, y, w, h])
                    elif len(parts) == 4:
                        # [x, y, w, h] - no class
                        x, y, w, h = map(float, parts[0:4])
                        boxes.append([x, y, w, h])
                    else:
                        print(f"Warning: Line {line_num} in {txt_file} has {len(parts)} parts, expected 4 or 5")
                        
        except Exception as e:
            print(f"Error reading file {txt_file}: {e}")
            continue
            
        labels_dict[filename] = boxes
    
    return labels_dict

def calculate_roc_auc(ground_truth_data, predictions_data, iou_threshold=0.5):
    """
    Calculate ROC AUC for object detection
    
    Parameters:
    - ground_truth_data: dict {image_id: [[x, y, w, h], ...]}
    - predictions_data: dict {image_id: [[x, y, w, h, confidence], ...]}
    - iou_threshold: threshold for considering detection as True Positive
    
    Returns:
    - roc_auc: ROC AUC score
    - all_scores: list of confidence scores
    - all_labels: list of binary labels (1 for TP, 0 for FP)
    """
    
    all_scores = []
    all_labels = []
    
    processed_images = 0
    total_predictions = 0
    total_gt_boxes = 0
    
    for image_id, gt_boxes in ground_truth_data.items():
        if image_id not in predictions_data:
            # Если для этого изображения нет предсказаний, пропускаем
            continue
            
        pred_boxes = predictions_data[image_id]
        total_gt_boxes += len(gt_boxes)
        
        # Sort predictions by confidence (descending)
        pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
        
        # Track which GT boxes have been matched
        gt_matched = [False] * len(gt_boxes)
        
        for pred_box in pred_boxes_sorted:
            x_pred, y_pred, w_pred, h_pred, confidence = pred_box
            pred_bbox = [x_pred, y_pred, w_pred, h_pred]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT box
            for i, gt_box in enumerate(gt_boxes):
                if not gt_matched[i]:
                    iou = calculate_iou(pred_bbox, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            # Determine if this prediction is TP or FP
            if best_iou >= iou_threshold and best_gt_idx != -1:
                all_scores.append(confidence)
                all_labels.append(1)  # True Positive
                gt_matched[best_gt_idx] = True
            else:
                all_scores.append(confidence)
                all_labels.append(0)  # False Positive
        
        processed_images += 1
        total_predictions += len(pred_boxes)
    
    print(f"Processed {processed_images} images")
    print(f"Total ground truth boxes: {total_gt_boxes}")
    print(f"Total predictions analyzed: {total_predictions}")
    
    # Calculate ROC AUC
    if len(all_scores) > 0:
        roc_auc = metrics.roc_auc_score(all_labels, all_scores)
        
        # Additional statistics
        true_positives = sum(all_labels)
        false_positives = len(all_labels) - true_positives
        
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"Positive Rate: {true_positives/len(all_scores):.3f}")
        
    else:
        roc_auc = 0.0
        print("No predictions to analyze!")
    
    return roc_auc, all_scores, all_labels

def analyze_dataset_statistics(ground_truth_data, predictions_data):
    """
    Analyze basic statistics about the dataset
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    # Ground Truth statistics
    total_gt_boxes = 0
    images_with_objects = 0
    images_without_objects = 0
    
    for image_id, boxes in ground_truth_data.items():
        total_gt_boxes += len(boxes)
        if len(boxes) > 0:
            images_with_objects += 1
        else:
            images_without_objects += 1
    
    print(f"Total images in ground truth: {len(ground_truth_data)}")
    print(f"Images with objects: {images_with_objects}")
    print(f"Images without objects: {images_without_objects}")
    print(f"Total ground truth boxes: {total_gt_boxes}")
    
    # Predictions statistics
    total_pred_boxes = 0
    for image_id, boxes in predictions_data.items():
        total_pred_boxes += len(boxes)
    
    print(f"Total images in predictions: {len(predictions_data)}")
    print(f"Total prediction boxes: {total_pred_boxes}")
    
    # Common images
    common_images = set(ground_truth_data.keys()) & set(predictions_data.keys())
    print(f"Common images (with both GT and predictions): {len(common_images)}")
    
    # Images only in GT
    only_in_gt = set(ground_truth_data.keys()) - set(predictions_data.keys())
    if only_in_gt:
        print(f"Images only in ground truth: {len(only_in_gt)}")
    
    # Images only in predictions
    only_in_pred = set(predictions_data.keys()) - set(ground_truth_data.keys())
    if only_in_pred:
        print(f"Images only in predictions: {len(only_in_pred)}")

def plot_confidence_distribution(scores, labels, iou_threshold):
    """Plot distribution of confidence scores for TP and FP"""
    if len(scores) == 0:
        print("No data to plot")
        return
        
    tp_scores = [score for score, label in zip(scores, labels) if label == 1]
    fp_scores = [score for score, label in zip(scores, labels) if label == 0]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if tp_scores:
        plt.hist(tp_scores, bins=20, alpha=0.7, label='True Positives', color='green', edgecolor='black')
    if fp_scores:
        plt.hist(fp_scores, bins=20, alpha=0.7, label='False Positives', color='red', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title(f'Confidence Distribution (IoU thr={iou_threshold})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Box plot
    data_to_plot = []
    labels_to_plot = []
    if tp_scores:
        data_to_plot.append(tp_scores)
        labels_to_plot.append('TP')
    if fp_scores:
        data_to_plot.append(fp_scores)
        labels_to_plot.append('FP')
    
    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels_to_plot)
        plt.title('Confidence Scores Boxplot')
        plt.ylabel('Confidence Score')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ground_truth_folder = "shuffle_test/labels"  # папка с разметкой
    predictions_folder = "runs/detect/predict/labels"    # папка с предсказаниями модели
    
    print("Loading ground truth labels...")
    ground_truth = load_labels_from_folder(ground_truth_folder, has_confidence=False)
    
    print("Loading prediction labels...")
    predictions = load_labels_from_folder(predictions_folder, has_confidence=True)
    
    # Анализ статистики датасета
    analyze_dataset_statistics(ground_truth, predictions)
    
    # Расчет ROC AUC с разными порогами IoU
    iou_thresholds = [0.3, 0.4, 0.5, 0.6]
    results = {}
    
    for iou_thresh in iou_thresholds:
        print(f"\n{'='*60}")
        print(f"Calculating ROC AUC with IoU threshold = {iou_thresh}")
        print(f"{'='*60}")
        
        roc_auc, scores, labels = calculate_roc_auc(
            ground_truth, 
            predictions, 
            iou_threshold=iou_thresh
        )
        
        results[iou_thresh] = {
            'roc_auc': roc_auc,
            'scores': scores,
            'labels': labels
        }
        
        print(f"ROC AUC: {roc_auc:.4f}")

        plot_confidence_distribution(scores, labels, iou_thresh)
        plot_roc_curve(scores, labels, roc_auc, iou_thresh)
        plot_comparative_roc_curves(results)
        
        # Визуализация для текущего порога
        plot_confidence_distribution(scores, labels, iou_thresh)
    
    # Сводная таблица результатов
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    print("IoU Threshold | ROC AUC")
    print("-" * 30)
    for iou_thresh in iou_thresholds:
        print(f"{iou_thresh:13.1f} | {results[iou_thresh]['roc_auc']:.4f}")