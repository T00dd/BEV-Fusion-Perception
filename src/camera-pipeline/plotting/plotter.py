import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Imposta un tema pulito per i grafici
sns.set_theme(style="whitegrid")

def plot_training_results(step_csv, epoch_csv, output_dir="plots"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    step_df = pd.read_csv(step_csv)
    epoch_df = pd.read_csv(epoch_csv)
    
    #Step Losses 
    plt.figure(figsize=(10, 6))
    window = max(1, len(step_df) // 50)  # Finestra dinamica
    
    plt.plot(step_df['global_step'], step_df['loss_total'].rolling(window).mean(), label='Total Loss (smoothed)', color='black', linewidth=2)
    plt.plot(step_df['global_step'], step_df['loss_focal'].rolling(window).mean(), label='Focal Loss (smoothed)', alpha=0.8)
    plt.plot(step_df['global_step'], step_df['loss_offset'].rolling(window).mean(), label='Offset Loss (smoothed)', alpha=0.8)
    
    plt.title('Training Losses over Steps (Smoothed)')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(out_dir / '1_step_losses.png', bbox_inches='tight', dpi=150)
    plt.close()

    #Learning Rates Schedule

    plt.figure(figsize=(10, 6))
    plt.plot(step_df['global_step'], step_df['lr_head'], label='LR Head', color='tab:orange')
    plt.plot(step_df['global_step'], step_df['lr_backbone'], label='LR Backbone', color='tab:blue')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Global Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.savefig(out_dir / '2_learning_rates.png', bbox_inches='tight', dpi=150)
    plt.close()

    #Epoch Losses (Train vs Validation)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total Loss
    axs[0].plot(epoch_df['epoch'], epoch_df['train_loss_total'], label='Train', color='tab:blue', marker='.')
    if 'val_loss_total' in epoch_df:
        val_df = epoch_df.dropna(subset=['val_loss_total'])
        axs[0].plot(val_df['epoch'], val_df['val_loss_total'], label='Validation', color='tab:orange', marker='o')
    axs[0].set_title('Total Loss per Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    # Focal Loss
    axs[1].plot(epoch_df['epoch'], epoch_df['train_loss_focal'], label='Train', color='tab:blue', marker='.')
    if 'val_loss_focal' in epoch_df:
        val_df = epoch_df.dropna(subset=['val_loss_focal'])
        axs[1].plot(val_df['epoch'], val_df['val_loss_focal'], label='Validation', color='tab:orange', marker='o')
    axs[1].set_title('Focal Loss per Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    # Offset Loss
    axs[2].plot(epoch_df['epoch'], epoch_df['train_loss_offset'], label='Train', color='tab:blue', marker='.')
    if 'val_loss_offset' in epoch_df:
        val_df = epoch_df.dropna(subset=['val_loss_offset'])
        axs[2].plot(val_df['epoch'], val_df['val_loss_offset'], label='Validation', color='tab:orange', marker='o')
    axs[2].set_title('Offset Loss per Epoch')
    axs[2].set_xlabel('Epoch')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(out_dir / '3_epoch_losses.png', bbox_inches='tight', dpi=150)
    plt.close()

    #Validation Metrics (F1, Precision, Recall)

    if 'val_f1' in epoch_df.columns:
        plt.figure(figsize=(10, 6))
        val_df = epoch_df.dropna(subset=['val_f1'])
        plt.plot(val_df['epoch'], val_df['val_f1'], label='F1 Score', marker='o', linewidth=2, color='tab:green')
        plt.plot(val_df['epoch'], val_df['val_precision'], label='Precision', marker='x', linestyle='--', color='tab:blue')
        plt.plot(val_df['epoch'], val_df['val_recall'], label='Recall', marker='s', linestyle='-.', color='tab:orange')
        
        plt.title('Validation Metrics (F1, Precision, Recall)')
        plt.xlabel('Epoch')
        plt.ylabel('Score (0 to 1)')
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.savefig(out_dir / '4_val_metrics.png', bbox_inches='tight', dpi=150)
        plt.close()

    #Validation Counts (TP, FP, FN)
    if 'val_tp' in epoch_df.columns:
        plt.figure(figsize=(10, 6))
        val_df = epoch_df.dropna(subset=['val_tp'])
        plt.plot(val_df['epoch'], val_df['val_tp'], label='True Positives (TP)', marker='o', color='tab:green')
        plt.plot(val_df['epoch'], val_df['val_fp'], label='False Positives (FP)', marker='x', color='tab:red')
        plt.plot(val_df['epoch'], val_df['val_fn'], label='False Negatives (FN)', marker='s', color='tab:orange')
        plt.yscale('log')
        plt.title('Validation Detections Count')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Objects')
        plt.legend()
        plt.savefig(out_dir / '5_val_counts.png', bbox_inches='tight', dpi=150)
        plt.close()

    #validation Recall by Distance (Fasce di Distanza)

    # Definiamo le colonne estratte da metrics.py e i colori associati (dal verde al rosso)
    distance_bins = [
        ('val_recall_0-5m', '0-5m', 'tab:green'),
        ('val_recall_5-10m', '5-10m', 'tab:blue'),
        ('val_recall_10-15m', '10-15m', 'tab:purple'),
        ('val_recall_15-20m', '15-20m', 'tab:orange'),
        ('val_recall_20-30m', '20-30m', 'tab:red'),
        ('val_recall_30-50m', '30-50m', 'darkred')
    ]
    
    # Controlliamo che almeno la prima colonna esista per evitare crash
    if distance_bins[0][0] in epoch_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Filtriamo via i NaN (le epoche in cui non facciamo validation)
        val_df = epoch_df.dropna(subset=[distance_bins[0][0]])
        
        for col_name, label, color in distance_bins:
            if col_name in val_df.columns:
                plt.plot(val_df['epoch'], val_df[col_name], label=label, marker='o', color=color, markersize=5)
                
        plt.title('Validation Recall by Distance Range')
        plt.xlabel('Epoch')
        plt.ylabel('Recall (0 to 1)')
        plt.ylim(-0.05, 1.05)
        plt.legend(title="Distance Bins", loc='lower right')
        plt.savefig(out_dir / '6_val_recall_by_distance.png', bbox_inches='tight', dpi=150)
        plt.close()

    #average precision (AP) & color accuracy
    if 'val_ap' in epoch_df.columns or 'val_color_accuracy' in epoch_df.columns:
        plt.figure(figsize=(10, 6))
        
        if 'val_ap' in epoch_df.columns:
            val_df_ap = epoch_df.dropna(subset=['val_ap'])
            plt.plot(val_df_ap['epoch'], val_df_ap['val_ap'], label='Average Precision (AP)', marker='o', color='tab:green', linewidth=2)
            
        if 'val_color_accuracy' in epoch_df.columns:
            val_df_col = epoch_df.dropna(subset=['val_color_accuracy'])
            plt.plot(val_df_col['epoch'], val_df_col['val_color_accuracy'], label='Color Accuracy (Overall)', marker='s', color='tab:blue', linewidth=2, linestyle='--')

        plt.title('Validation AP and Color Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Score (0 to 1)')
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='lower right')
        plt.savefig(out_dir / '7_val_ap_color_acc.png', bbox_inches='tight', dpi=150)
        plt.close()

    #per-class F1 score (blue, yellow, orange)
    class_f1_cols = [
        ('val_f1_c0', 'Blue Cones (c0)', '#2b6cb0'),
        ('val_f1_c1', 'Yellow Cones (c1)', '#d4a017'),
        ('val_f1_c2', 'Orange Cones (c2)', '#dd6b20')
    ]
    if any(col[0] in epoch_df.columns for col in class_f1_cols):
        plt.figure(figsize=(10, 6))
        first_col = next(col[0] for col in class_f1_cols if col[0] in epoch_df.columns)
        val_df = epoch_df.dropna(subset=[first_col])

        for col_name, label, color in class_f1_cols:
            if col_name in val_df.columns:
                plt.plot(val_df['epoch'], val_df[col_name], label=label, marker='o', color=color, linewidth=2)

        plt.title('Per-Class F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score (0 to 1)')
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='lower right')
        plt.savefig(out_dir / '8_val_per_class_f1.png', bbox_inches='tight', dpi=150)
        plt.close()

    #color confusion errors (blue as yellow vs yellow as blue)
    if 'val_color_acc_c0' in epoch_df.columns:
        val_df = epoch_df.dropna(subset=['val_color_acc_c0'])
        if not val_df.empty:
            plt.figure(figsize=(8, 6))
            last_row = val_df.iloc[-1]
            epoch_num = int(last_row['epoch'])

            # Estrazione accuratezze diagonale
            c0_acc = last_row.get('val_color_acc_c0', 0.0)
            c1_acc = last_row.get('val_color_acc_c1', 0.0)
            c2_acc = last_row.get('val_color_acc_c2', 0.0)

            # Estrazione tot coni GT per classe (se presenti nei log)
            gt_c0 = last_row.get('val_num_gt_c0', 100)
            gt_c1 = last_row.get('val_num_gt_c1', 100)

            b_as_y = last_row.get('val_confusion_blue_as_yellow', 0)
            y_as_b = last_row.get('val_confusion_yellow_as_blue', 0)

            # Costruzione matrice normalizzata per riga (GT)
            cm_norm = np.zeros((3, 3))
            
            # Riga 0: GT Blue
            cm_norm[0, 0] = c0_acc
            cm_norm[0, 1] = b_as_y / max(gt_c0, 1)
            cm_norm[0, 2] = max(0.0, 1.0 - cm_norm[0, 0] - cm_norm[0, 1])

            # Riga 1: GT Yellow
            cm_norm[1, 0] = y_as_b / max(gt_c1, 1)
            cm_norm[1, 1] = c1_acc
            cm_norm[1, 2] = max(0.0, 1.0 - cm_norm[1, 0] - cm_norm[1, 1])

            # Riga 2: GT Orange
            cm_norm[2, 2] = c2_acc
            rem_c2 = max(0.0, 1.0 - c2_acc)
            cm_norm[2, 0] = rem_c2 / 2.0
            cm_norm[2, 1] = rem_c2 / 2.0

            class_names = ['Blue', 'Yellow', 'Orange']
            
            # Plot Heatmap
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        vmin=0, vmax=1, cbar=True, annot_kws={"size": 11, "weight": "bold"})

            plt.title(f'Color Confusion Matrix (Normalized) - Epoch {epoch_num:03d}', fontsize=12)
            plt.xlabel('Predicted Class', fontsize=11)
            plt.ylabel('Ground Truth Class', fontsize=11)
            plt.savefig(out_dir / '9_val_confusion_matrix.png', bbox_inches='tight', dpi=150)
            plt.close()

    print(f"I grafici sono stati generati e salvati in '{out_dir}/'")
if __name__ == "__main__":
    plot_training_results('../checkpoints/bev/step_log.csv', '../checkpoints/bev/epoch_log.csv', output_dir='plots')