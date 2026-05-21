"""
plotting.py — Tema visual y funciones reutilizables para el notebook integrador.

Separa la configuración estética (colores, fuentes, layout) de la lógica de datos.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ─── Paleta y constantes ─────────────────────────────────────────────────

C = {
    'train':     '#2563eb',
    'val':       '#ef4444',
    'normal':    '#3b82f6',
    'pneumonia': '#ef4444',
    'accent':    '#10b981',
    'warn':      '#f59e0b',
    'purple':    '#8b5cf6',
    'dark':      '#1e293b',
    'muted':     '#94a3b8',
    'grid':      '#e2e8f0',
    'bg':        '#f8fafc',
    'white':     '#ffffff',
    'border':    '#cbd5e1',
    'correct':   '#059669',
    'incorrect': '#dc2626',
}

MODEL_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']

CLASS_NAMES = {0: 'NORMAL', 1: 'PNEUMONIA'}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor':   C['bg'],
        'axes.facecolor':     C['white'],
        'axes.edgecolor':     C['border'],
        'axes.linewidth':     0.8,
        'axes.grid':          True,
        'grid.alpha':         0.25,
        'grid.color':         C['grid'],
        'grid.linewidth':     0.6,
        'axes.titleweight':   'bold',
        'axes.titlesize':     12,
        'axes.titlepad':      10,
        'axes.labelsize':     10,
        'axes.labelpad':      6,
        'axes.labelcolor':    C['dark'],
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,
        'xtick.color':        C['muted'],
        'ytick.color':        C['muted'],
        'legend.fontsize':    9,
        'legend.framealpha':  0.95,
        'legend.edgecolor':   C['grid'],
        'legend.borderpad':   0.6,
        'font.family':        'sans-serif',
        'figure.dpi':         110,
        'savefig.dpi':        150,
        'figure.titlesize':   14,
        'figure.titleweight': 'bold',
    })


setup_style()


# ─── Helpers internos ────────────────────────────────────────────────────

def _finish(fig, tight=True):
    if tight:
        fig.tight_layout()
    plt.show()


def _annotate_bars(ax, bars, fmt='{:.4f}', offset=0.02, **kw):
    defaults = dict(ha='center', va='bottom', fontsize=9, fontweight='bold', color=C['dark'])
    defaults.update(kw)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset, fmt.format(h), **defaults)


# ─── Curvas de entrenamiento ─────────────────────────────────────────────

def plot_training_curves(history, title='', ylim_acc=(0.4, 1.0)):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    ep = range(1, len(history['train_loss']) + 1)

    axes[0].plot(ep, history['train_loss'], label='Train', color=C['train'], lw=2)
    axes[0].plot(ep, history['val_loss'],   label='Val',   color=C['val'],   lw=2, ls='--')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Pérdida por Época')
    axes[0].legend()

    axes[1].plot(ep, history['train_acc'], label='Train', color=C['train'], lw=2)
    axes[1].plot(ep, history['val_acc'],   label='Val',   color=C['val'],   lw=2, ls='--')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy por Época')
    axes[1].set_ylim(ylim_acc)
    axes[1].legend()

    _finish(fig)

    print(f"  Train — loss: {history['train_loss'][-1]:.4f}  acc: {history['train_acc'][-1]:.4f}")
    print(f"  Val   — loss: {history['val_loss'][-1]:.4f}  acc: {history['val_acc'][-1]:.4f}")


# ─── Curvas de entrenamiento con accuracy bar ────────────────────────────

def plot_training_curves_with_bar(history, title=''):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    ep = range(1, len(history['losses']) + 1)

    axes[0].plot(ep, history['losses'], lw=2, color=C['train'])
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Curva de Pérdida')

    axes[1].plot(ep, history['train_accuracies'], label='Train', color=C['train'], lw=2)
    axes[1].plot(ep, history['val_accuracies'],   label='Val',   color=C['val'],   lw=2, ls='--')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy por Época')
    axes[1].set_ylim([0.4, 1.0])
    axes[1].legend()

    final_train = history['train_accuracies'][-1]
    final_val   = history['val_accuracies'][-1]
    bars = axes[2].bar(['Train', 'Val'], [final_train, final_val],
                       color=[C['train'], C['val']], alpha=0.85, edgecolor=C['dark'], lw=1.2,
                       width=0.5)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Accuracy Final')
    axes[2].set_ylim([0, 1.0])
    axes[2].grid(axis='x', visible=False)
    _annotate_bars(axes[2], bars)

    _finish(fig)

    print(f"  Accuracy final — Train: {final_train:.4f}  Val: {final_val:.4f}")


# ─── Confusion matrix + métricas derivadas ───────────────────────────────

def plot_confusion_and_metrics(y_true, y_pred, title=''):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'],
                annot_kws={'size': 16, 'weight': 'bold'},
                linewidths=2, linecolor=C['white'])
    axes[0].set_xlabel('Predicción', fontweight='bold')
    axes[0].set_ylabel('Etiqueta Real', fontweight='bold')
    axes[0].set_title('Matriz de Confusión')

    names  = ['Sensibilidad\n(Recall)', 'Especificidad', 'Tasa FP', 'Tasa FN']
    values = [sensitivity, specificity, fpr, fnr]
    colors = [C['accent'], C['train'], C['val'], C['warn']]

    bars = axes[1].bar(names, values, color=colors, alpha=0.85, edgecolor=C['dark'], lw=1.2, width=0.55)
    axes[1].set_ylabel('Valor')
    axes[1].set_title('Métricas Derivadas')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='x', visible=False)
    _annotate_bars(axes[1], bars)

    _finish(fig)

    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Sensibilidad: {sensitivity:.4f}   Especificidad: {specificity:.4f}")
    print(f"  Tasa FP: {fpr:.4f}   Tasa FN: {fnr:.4f}")


# ─── Confusion matrix + ROC ──────────────────────────────────────────────

def plot_confusion_and_roc(y_true, y_pred, y_proba, title=''):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'],
                annot_kws={'size': 16, 'weight': 'bold'},
                linewidths=2, linecolor=C['white'])
    axes[0].set_xlabel('Predicción', fontweight='bold')
    axes[0].set_ylabel('Etiqueta Real', fontweight='bold')
    axes[0].set_title('Matriz de Confusión')

    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_proba)
    axes[1].plot(fpr_curve, tpr_curve, color=C['warn'], lw=2.5, label=f'ROC (AUC = {auc:.4f})')
    axes[1].plot([0, 1], [0, 1], ls='--', lw=1, color=C['muted'], label='Aleatorio')
    axes[1].fill_between(fpr_curve, tpr_curve, alpha=0.08, color=C['warn'])
    axes[1].set_xlabel('Tasa de Falsos Positivos', fontweight='bold')
    axes[1].set_ylabel('Tasa de Verdaderos Positivos', fontweight='bold')
    axes[1].set_title('Curva ROC')
    axes[1].legend(loc='lower right')

    _finish(fig)

    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Sensibilidad: {tp/(tp+fn):.4f}   Especificidad: {tn/(tn+fp):.4f}")
    print(f"  AUC-ROC: {auc:.4f}")


# ─── Muestras de imágenes ────────────────────────────────────────────────

def plot_image_samples(images_normal, images_pneumonia, title='Muestras: NORMAL vs PNEUMONIA'):
    n = max(len(images_normal), len(images_pneumonia))
    fig, axes = plt.subplots(2, n, figsize=(min(n * 1.6, 18), 4))
    fig.suptitle(title, fontsize=13)

    for idx in range(n):
        for row, (imgs, label) in enumerate([(images_normal, 'NORMAL'), (images_pneumonia, 'PNEUMONIA')]):
            ax = axes[row, idx] if n > 1 else axes[row]
            if idx < len(imgs):
                ax.imshow(imgs[idx], cmap='gray')
                ax.set_title(f'{label} {idx+1}', fontsize=7, color=C['normal'] if row == 0 else C['pneumonia'])
            ax.axis('off')

    _finish(fig)


# ─── Histogramas de píxeles ──────────────────────────────────────────────

def plot_pixel_histograms(normal_data, pneumonia_data, normal_stats, pneumonia_stats):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    fig.suptitle('Distribución de Valores de Píxeles', fontsize=13, y=1.02)

    normal_flat = normal_data.flatten()
    pneumonia_flat = pneumonia_data.flatten()

    for ax, data, stats, label, color in [
        (axes[0], normal_flat,    normal_stats,    'NORMAL',    C['normal']),
        (axes[1], pneumonia_flat, pneumonia_stats, 'PNEUMONIA', C['pneumonia']),
    ]:
        ax.hist(data, bins=50, alpha=0.75, color=color, edgecolor='white', lw=0.5)
        ax.axvline(stats['mean'], color=C['dark'], ls='--', lw=1.8, label=f"μ={stats['mean']:.3f}")
        ax.axvline(stats['mean'] - stats['std'], color=C['warn'], ls=':', lw=1.5, label='±σ')
        ax.axvline(stats['mean'] + stats['std'], color=C['warn'], ls=':', lw=1.5)
        ax.set_title(label)
        ax.set_xlabel('Valor de píxel [0, 1]')
        ax.set_ylabel('Frecuencia')
        ax.legend(fontsize=8)

    axes[2].hist(normal_flat,    bins=50, alpha=0.55, label='NORMAL',    color=C['normal'],    edgecolor='white', lw=0.3)
    axes[2].hist(pneumonia_flat, bins=50, alpha=0.55, label='PNEUMONIA', color=C['pneumonia'], edgecolor='white', lw=0.3)
    axes[2].set_title('Superpuesto')
    axes[2].set_xlabel('Valor de píxel [0, 1]')
    axes[2].set_ylabel('Frecuencia')
    axes[2].legend()

    _finish(fig)


# ─── Scatter + regresión ─────────────────────────────────────────────────

def plot_regression_scatter(datasets, model, title=''):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2))
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    for idx, (X_feat, y, label) in enumerate(datasets):
        ax = axes[idx] if len(datasets) > 1 else axes
        ax.scatter(X_feat[y == 0], y[y == 0], alpha=0.45, label='NORMAL',    color=C['normal'],    s=25, edgecolors='white', lw=0.3)
        ax.scatter(X_feat[y == 1], y[y == 1], alpha=0.45, label='PNEUMONIA', color=C['pneumonia'], s=25, edgecolors='white', lw=0.3)

        x_range = np.array([X_feat.min(), X_feat.max()]).reshape(-1, 1)
        y_line = model.predict(x_range)
        ax.plot(x_range, y_line, color=C['accent'], lw=2.5,
                label=f'y = {model.w:.4f}x + {model.b:.4f}')
        ax.axhline(y=0.5, color=C['muted'], ls='--', alpha=0.6, lw=1, label='Umbral 0.5')

        ax.set_xlabel('Intensidad Media de Píxel')
        ax.set_ylabel('Etiqueta')
        r2 = model.score(X_feat, y.astype(np.float32))
        ax.set_title(f'{label} (R² = {r2:.4f})')
        ax.set_ylim(-0.3, 1.3)
        ax.legend(fontsize=7, loc='best')

    _finish(fig)


# ─── Comparación de regularizaciones ─────────────────────────────────────

def plot_regularization_comparison(reg_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle('Comparación de Técnicas de Regularización', fontsize=13, y=1.02)

    for i, (name, data) in enumerate(reg_results.items()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ep = range(1, len(data['hist']['val_loss']) + 1)
        axes[0].plot(ep, data['hist']['val_loss'], label=name, color=color, lw=1.8)
        axes[1].plot(ep, data['hist']['val_acc'],  label=name, color=color, lw=1.8)

    axes[0].set_title('Val Loss')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend(fontsize=8)

    axes[1].set_title('Val Accuracy')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0.4, 1.0])
    axes[1].legend(fontsize=8)

    _finish(fig)


# ─── Comparación de optimizadores ────────────────────────────────────────

def plot_optimizer_comparison(opt_results):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle('Comparación de Optimizadores — Train vs Val Loss', fontsize=13, y=1.02)

    for i, (name, data) in enumerate(opt_results.items()):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ep = range(1, len(data['hist']['train_loss']) + 1)
        ax.plot(ep, data['hist']['train_loss'], ls='--', alpha=0.5, color=color, lw=1.5)
        ax.plot(ep, data['hist']['val_loss'],   label=f'{name}', color=color, lw=2)

    ax.set_xlabel('Época')
    ax.set_ylabel('BCE Loss')
    ax.legend(fontsize=9)

    _finish(fig)


# ─── Feature maps con kernels manuales ───────────────────────────────────

def plot_kernel_feature_maps(img, feature_maps, kernel_names, class_name=''):
    n = 1 + len(feature_maps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if class_name:
        fig.suptitle(f'Mapas de Características — {class_name}', fontsize=13, y=1.02)

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original (64×64)')
    axes[0].axis('off')

    for i, (fm, name) in enumerate(zip(feature_maps, kernel_names)):
        axes[i + 1].imshow(fm, cmap='gray')
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')

    _finish(fig)


# ─── Activaciones CNN por bloque ─────────────────────────────────────────

def plot_cnn_activations(img_tensor, activations, label='', n_cols=8):
    n_blocks = len(activations)
    fig, axes = plt.subplots(1 + n_blocks, n_cols, figsize=(n_cols * 2, (1 + n_blocks) * 2))
    fig.suptitle(f'Activaciones CNN — {label}', fontsize=13, y=1.01)

    axes[0][0].imshow(img_tensor.cpu().squeeze(), cmap='gray')
    axes[0][0].set_title('Original', fontsize=8)
    for j in range(n_cols):
        axes[0][j].axis('off')

    for row, (block_name, feat_maps) in enumerate(activations.items()):
        n_filters = feat_maps.shape[1]
        step = max(n_filters // n_cols, 1)
        for j in range(n_cols):
            idx = min(j * step, n_filters - 1)
            axes[row + 1][j].imshow(feat_maps[0, idx].numpy(), cmap='viridis')
            if j == 0:
                axes[row + 1][j].set_ylabel(block_name, fontsize=7, rotation=90, labelpad=30)
            axes[row + 1][j].set_title(f'F{idx}', fontsize=7)
            axes[row + 1][j].axis('off')

    _finish(fig)


# ─── Predicciones correctas vs incorrectas ───────────────────────────────

def plot_predictions(X_test, y_true, y_pred, y_proba, n=5, title='Predicciones del Modelo'):
    correct_idx   = np.where(y_pred == y_true)[0]
    incorrect_idx = np.where(y_pred != y_true)[0]

    np.random.seed(42)
    sample_correct   = np.random.choice(correct_idx,   min(n, len(correct_idx)),   replace=False)
    sample_incorrect = np.random.choice(incorrect_idx, min(n, len(incorrect_idx)), replace=False)

    fig, axes = plt.subplots(2, n, figsize=(n * 3.2, 7))
    fig.suptitle(title, fontsize=13, y=1.01)

    for col, idx in enumerate(sample_correct):
        img = X_test[idx, :, :, 0] if X_test.ndim == 4 else X_test[idx]
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(
            f'Real: {CLASS_NAMES[int(y_true[idx])]}\n'
            f'Pred: {CLASS_NAMES[int(y_pred[idx])]} ({y_proba[idx]:.2f})',
            fontsize=8, color=C['correct'])
        axes[0, col].axis('off')
    axes[0, 0].set_ylabel('CORRECTAS', fontsize=11, fontweight='bold', rotation=90, labelpad=5)

    for col, idx in enumerate(sample_incorrect):
        img = X_test[idx, :, :, 0] if X_test.ndim == 4 else X_test[idx]
        axes[1, col].imshow(img, cmap='gray')
        axes[1, col].set_title(
            f'Real: {CLASS_NAMES[int(y_true[idx])]}\n'
            f'Pred: {CLASS_NAMES[int(y_pred[idx])]} ({y_proba[idx]:.2f})',
            fontsize=8, color=C['incorrect'])
        axes[1, col].axis('off')
    axes[1, 0].set_ylabel('INCORRECTAS', fontsize=11, fontweight='bold', rotation=90, labelpad=5)

    _finish(fig)

    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]
    print(f"  Falsos Positivos: {len(fp_idx)}   Falsos Negativos: {len(fn_idx)}")
    if len(fp_idx) > 0:
        print(f"  Prob. media FP: {y_proba[fp_idx].mean():.4f}")
    if len(fn_idx) > 0:
        print(f"  Prob. media FN: {y_proba[fn_idx].mean():.4f}")


# ─── ResNet activaciones antes/después ───────────────────────────────────

def plot_resnet_before_after(acts_before, acts_after, prob_before, prob_after,
                              layer_names, n_filters=4):
    fig, axes = plt.subplots(len(layer_names), 2 * n_filters + 1, figsize=(18, 3 * len(layer_names)),
                              gridspec_kw={'width_ratios': [1]*n_filters + [0.15] + [1]*n_filters})
    fig.suptitle(
        f'Activaciones ResNet-18 — Antes vs Después del Fine-Tuning\n'
        f'Antes: P(PNEUMONIA)={prob_before:.3f}  |  Después: P(PNEUMONIA)={prob_after:.3f}',
        fontsize=12, y=1.03)

    for row, layer_name in enumerate(layer_names):
        feat_before = acts_before[layer_name][0]
        feat_after  = acts_after[layer_name][0]

        var_after = feat_after.var(dim=(1, 2))
        top_idx = var_after.argsort(descending=True)[:n_filters]

        for j, fi in enumerate(top_idx):
            ax_b = axes[row][j]
            ax_b.imshow(feat_before[fi].numpy(), cmap='inferno')
            ax_b.axis('off')
            if row == 0:
                ax_b.set_title(f'F{fi.item()}', fontsize=8)
            if j == 0:
                ax_b.set_ylabel(layer_name, fontsize=9, fontweight='bold', rotation=90, labelpad=35)

        axes[row][n_filters].axis('off')

        for j, fi in enumerate(top_idx):
            ax_a = axes[row][n_filters + 1 + j]
            ax_a.imshow(feat_after[fi].numpy(), cmap='inferno')
            ax_a.axis('off')
            if row == 0:
                ax_a.set_title(f'F{fi.item()}', fontsize=8)

    fig.text(0.27, 0.96, 'ANTES (ImageNet)', ha='center', fontsize=11, style='italic', color=C['val'])
    fig.text(0.73, 0.96, 'DESPUÉS (Neumonía)', ha='center', fontsize=11, style='italic', color=C['accent'])

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ─── Overfitting: train vs val ───────────────────────────────────────────

def plot_overfitting(history, title='Overfitting: Train vs Val (sin regularización)'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.suptitle(title, fontsize=13, y=1.02)

    ep = range(1, len(history['train_loss']) + 1)

    axes[0].plot(ep, history['train_loss'], label='Train', color=C['train'], lw=2)
    axes[0].plot(ep, history['val_loss'],   label='Val',   color=C['val'],   lw=2, ls='--')
    gap_start = next((i for i in range(len(history['val_loss']))
                      if history['val_loss'][i] - history['train_loss'][i] > 0.05), None)
    if gap_start is not None:
        ax0_y_max = max(max(history['train_loss']), max(history['val_loss']))
        axes[0].axvspan(gap_start, len(history['train_loss']), alpha=0.06, color=C['val'])
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Pérdida')
    axes[0].legend()

    axes[1].plot(ep, history['train_acc'], label='Train', color=C['train'], lw=2)
    axes[1].plot(ep, history['val_acc'],   label='Val',   color=C['val'],   lw=2, ls='--')
    if gap_start is not None:
        axes[1].axvspan(gap_start, len(history['train_acc']), alpha=0.06, color=C['val'])
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_ylim([0.4, 1.0])
    axes[1].legend()

    _finish(fig)

    gap = history['train_acc'][-1] - history['val_acc'][-1]
    print(f"  Brecha Train-Val: {gap:.4f}")
    print(f"  Train acc: {history['train_acc'][-1]:.4f}  Val acc: {history['val_acc'][-1]:.4f}")
