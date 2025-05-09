# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import seaborn as sns
import itertools

def plot_time_series(time_series, title='Time Series', figsize=(12, 6), save_path=None):
    """
    Zaman serisini çizdir.
    
    Args:
        time_series (pd.Series): Çizdirilecek zaman serisi
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    plt.figure(figsize=figsize)
    plt.plot(time_series.index, time_series.values)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # X eksenini tarih olarak biçimlendir
    if isinstance(time_series.index, pd.DatetimeIndex):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_seasonality_components(time_series, period=None, title='STL Decomposition', figsize=(12, 10), save_path=None):
    """
    Zaman serisinin bileşenlerini STL ayrıştırması ile çizdir.
    
    Args:
        time_series (pd.Series): Çizdirilecek zaman serisi
        period (int, optional): Mevsimsel dönem. None ise en uygun değer seçilir.
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    # En uygun periyot belirleme
    if period is None:
        potential_periods = [7, 14, 30, 90, 365]
        period = min([p for p in potential_periods if p < len(time_series) * 0.33], 
                     default=min(14, len(time_series) // 3))
    
    # STL ayrıştırması
    stl = STL(time_series, seasonal=period, robust=True)
    result = stl.fit()
    
    # Bileşenleri çizdir
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Orijinal seri
    axes[0].plot(time_series.index, time_series.values)
    axes[0].set_title('Original Series')
    
    # Trend bileşeni
    axes[1].plot(time_series.index, result.trend)
    axes[1].set_title('Trend Component')
    
    # Mevsimsel bileşen
    axes[2].plot(time_series.index, result.seasonal)
    axes[2].set_title(f'Seasonal Component (Period={period})')
    
    # Artık (Residual) bileşen
    axes[3].plot(time_series.index, result.resid)
    axes[3].set_title('Residual Component')
    
    # X eksenini tarih olarak biçimlendir
    if isinstance(time_series.index, pd.DatetimeIndex):
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.suptitle(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_time_series_with_breakpoints(time_series, breakpoints=None, title='Time Series with Breakpoints', figsize=(12, 6), save_path=None):
    """
    Kırılma noktalarıyla birlikte zaman serisini çizdir.
    
    Args:
        time_series (pd.Series): Çizdirilecek zaman serisi
        breakpoints (list): Kırılma noktalarının listesi
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    plt.figure(figsize=figsize)
    plt.plot(time_series.index, time_series.values)
    
    # Kırılma noktalarını çizdir
    if breakpoints:
        for bp in breakpoints:
            position = bp['position']
            if position < len(time_series):
                plt.axvline(x=time_series.index[position], color='r', linestyle='--', alpha=0.7)
                plt.text(time_series.index[position], time_series.max() * 1.1, 
                         f"Period: {bp['old_period']:.1f} → {bp['new_period']:.1f}", 
                         rotation=90, verticalalignment='bottom')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # X eksenini tarih olarak biçimlendir
    if isinstance(time_series.index, pd.DatetimeIndex):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', normalize=False, figsize=(10, 8), save_path=None):
    """
    Karışıklık matrisini çizdir.
    
    Args:
        y_true (array): Gerçek etiketler
        y_pred (array): Tahmin edilen etiketler
        classes (list): Sınıf isimleri
        title (str): Grafik başlığı
        normalize (bool): Normalize edilip edilmeyeceği
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_feature_importance(feature_names, importances, title='Feature Importance', figsize=(12, 8), save_path=None):
    """
    Özellik önemini çizdir.
    
    Args:
        feature_names (list): Özellik isimleri
        importances (array): Özellik önem değerleri
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    # Özellik önem değerlerine göre sırala
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # En önemli 20 özelliği al
    top_n = min(20, len(sorted_feature_names))
    
    plt.figure(figsize=figsize)
    plt.bar(range(top_n), sorted_importances[:top_n], align='center')
    plt.xticks(range(top_n), sorted_feature_names[:top_n], rotation=90)
    plt.title(title)
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_roc_curve(y_true, y_score, title='ROC Curve', figsize=(10, 8), save_path=None):
    """
    ROC eğrisini çizdir.
    
    Args:
        y_true (array): Gerçek etiketler
        y_score (array): Tahmin skorları
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_precision_recall_curve(y_true, y_score, title='Precision-Recall Curve', figsize=(10, 8), save_path=None):
    """
    Hassasiyet-Duyarlılık eğrisini çizdir.
    
    Args:
        y_true (array): Gerçek etiketler
        y_score (array): Tahmin skorları
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_model_comparison(model_names, metrics, metric_names, title='Model Comparison', figsize=(12, 8), save_path=None):
    """
    Model karşılaştırma grafiğini çizdir.
    
    Args:
        model_names (list): Model isimleri
        metrics (array): Model metrik değerleri (model x metrik)
        metric_names (list): Metrik isimleri
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    x = np.arange(len(model_names))
    width = 0.8 / len(metric_names)
    
    plt.figure(figsize=figsize)
    
    for i, metric_name in enumerate(metric_names):
        plt.bar(x + i * width - 0.4 + width/2, metrics[:, i], width, label=metric_name)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_multiple_time_series(time_series_list, titles=None, max_cols=3, figsize=(16, 16), save_path=None):
    """
    Birden fazla zaman serisini alt grafikler olarak çizdir.
    
    Args:
        time_series_list (list): Zaman serisi listesi
        titles (list, optional): Grafik başlıkları listesi
        max_cols (int): Maksimum sütun sayısı
        figsize (tuple): Grafik boyutu
        save_path (str, optional): Kaydedilecek dosya yolu
    """
    n_series = len(time_series_list)
    n_cols = min(max_cols, n_series)
    n_rows = (n_series + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Tek boyutlu dizi olmaması için
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    for i, ts in enumerate(time_series_list):
        if i < len(axes):
            ax = axes.flatten()[i] if hasattr(axes, 'flatten') else axes[i]
            ax.plot(ts.index, ts.values)
            
            # Başlık ekle
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
            else:
                ax.set_title(f'Series {i+1}')
            
            # X eksenini tarih olarak biçimlendir
            if isinstance(ts.index, pd.DatetimeIndex):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
                
            ax.grid(True, alpha=0.3)
    
    # Kullanılmayan alt grafikleri gizle
    for i in range(n_series, n_rows * n_cols):
        if i < len(axes.flatten()):
            axes.flatten()[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()