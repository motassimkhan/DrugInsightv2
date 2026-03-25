"""
Generate training curves chart for DrugInsight IEEE paper.
Run: python generate_training_curves.py
Output: training_curves.png (upload to Overleaf)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs('figs', exist_ok=True)

# ── Epoch data from training logs ──
epochs     = np.arange(1, 11)
train_loss = [0.6556, 0.6282, 0.6200, 0.6156, 0.6134, 0.6120, 0.6113, 0.6112, 0.6047, 0.6055]
val_auc    = [0.6945, 0.7055, 0.7002, 0.7065, 0.7001, 0.7047, 0.7042, 0.7041, 0.7023, 0.6996]
val_ap     = [0.6918, 0.6986, 0.6970, 0.7022, 0.6986, 0.7024, 0.7021, 0.6967, 0.6987, 0.6965]
train_acc  = [0.6066, 0.6474, 0.6574, 0.6626, 0.6656, 0.6670, 0.6685, 0.6688, 0.6754, 0.6748]
val_acc    = [0.6394, 0.6489, 0.6442, 0.6485, 0.6480, 0.6429, 0.6455, 0.6422, 0.6446, 0.6411]

best_epoch = 4  # 1-indexed

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

# ════════════════════════════════════════════════════════════
# Figure 1: Training Loss + Validation AUC (dual y-axis)
# ════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(7, 4))

color_loss = '#E74C3C'
color_auc  = '#2E86C1'

# Training Loss (left y-axis)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Loss', color=color_loss, fontsize=12, fontweight='bold')
line1, = ax1.plot(epochs, train_loss, 'o--', color=color_loss, linewidth=2,
                  markersize=6, label='Training Loss', alpha=0.9)
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_ylim(0.59, 0.67)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Validation AUC (right y-axis)
ax2 = ax1.twinx()
ax2.set_ylabel('Validation AUC / AP', color=color_auc, fontsize=12, fontweight='bold')
line2, = ax2.plot(epochs, val_auc, 's-', color=color_auc, linewidth=2.5,
                  markersize=7, label='Val AUC', alpha=0.9)
line3, = ax2.plot(epochs, val_ap, '^-', color='#27AE60', linewidth=2,
                  markersize=6, label='Val AP', alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color_auc)
ax2.set_ylim(0.68, 0.72)

# Mark best epoch
ax2.axvline(x=best_epoch, color='#F39C12', linestyle=':', linewidth=2, alpha=0.7)
ax2.annotate(f'Best (Epoch {best_epoch})\nAUC = 0.7065',
             xy=(best_epoch, 0.7065), xytext=(best_epoch + 1.5, 0.715),
             fontsize=9, fontweight='bold', color='#F39C12',
             arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7', edgecolor='#F39C12'))

# Legend
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=9, framealpha=0.9)

ax1.set_title('DrugInsight Training Curves (Cold-Start Drug-Level Split)',
              fontsize=13, fontweight='bold', pad=12)
ax1.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig(os.path.join('figs', 'training_curves.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: figs/training_curves.png")
plt.close()

# ════════════════════════════════════════════════════════════
# Figure 2: Accuracy Comparison
# ════════════════════════════════════════════════════════════
fig2, ax3 = plt.subplots(figsize=(7, 4))

ax3.plot(epochs, train_acc, 'o-', color='#8E44AD', linewidth=2, markersize=6, label='Train Accuracy')
ax3.plot(epochs, val_acc, 's-', color='#2E86C1', linewidth=2, markersize=6, label='Val Accuracy')

ax3.axvline(x=best_epoch, color='#F39C12', linestyle=':', linewidth=2, alpha=0.7,
            label=f'Best Model (Epoch {best_epoch})')

ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Training vs Validation Accuracy', fontsize=13, fontweight='bold', pad=12)
ax3.legend(fontsize=9, framealpha=0.9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.58, 0.70)
ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

fig2.tight_layout()
plt.savefig(os.path.join('figs', 'accuracy_curves.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: figs/accuracy_curves.png")
plt.close()

# ════════════════════════════════════════════════════════════
# Figure 3: Confusion Matrix for Best Epoch (Epoch 4)
# ════════════════════════════════════════════════════════════
from matplotlib.colors import LinearSegmentedColormap

# Epoch 4 confusion matrix
cm = np.array([[34148, 18410],
               [18538, 34020]])
labels_cm = ['Non-Interacting', 'Interacting']

fig3, ax4 = plt.subplots(figsize=(5, 4))
cmap = LinearSegmentedColormap.from_list('custom', ['#EBF5FB', '#2E86C1'])
im = ax4.imshow(cm, interpolation='nearest', cmap=cmap)
ax4.figure.colorbar(im, ax=ax4, shrink=0.8)

ax4.set(xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels_cm, yticklabels=labels_cm,
        ylabel='True Label', xlabel='Predicted Label')
ax4.set_title(f'Confusion Matrix (Epoch {best_epoch}, Best Model)',
              fontsize=12, fontweight='bold', pad=10)

# Text annotations
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        ax4.text(j, i, f'{cm[i, j]:,}',
                 ha='center', va='center', fontsize=13, fontweight='bold',
                 color='white' if cm[i, j] > thresh else 'black')

fig3.tight_layout()
plt.savefig(os.path.join('figs', 'confusion_matrix.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: figs/confusion_matrix.png")
plt.close()

print("\nAll figures generated in 'figs' folder. Upload to Overleaf.")
