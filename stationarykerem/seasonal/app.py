# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Ana proje klasörünü modül yoluna ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Kendi modüllerimizi içe aktar
from utils.seasonality_detector import SeasonalityDetector

# Sayfayı yapılandır
st.set_page_config(
    page_title="Zaman Serisi Mevsimsellik Analizi",
    page_icon="📈",
    layout="wide"
)

# Başlık ve açıklama
st.title("📊 Zaman Serisi Mevsimsellik Analizi")
st.markdown("""
Bu uygulama, yüklediğiniz zaman serisi verilerinde mevsimsellik analizi yapar.
XGBoost ve LSTM modellerini kullanarak, zaman serisinin mevsimsel olup olmadığını ve
mevsimsel kırılma noktalarını tespit eder.
""")

# Kenar çubuğu - Analiz Ayarları
with st.sidebar:
    st.header("Analiz Ayarları")
    
    use_xgboost = st.checkbox("XGBoost Modelini Kullan", value=True)
    
    # PyTorch'un varlığını kontrol et
    try:
        import torch
        pytorch_available = True
    except ImportError:
        pytorch_available = False
    
    # LSTM checkbox'ı, eğer PyTorch yüklü değilse disable et
    use_lstm = st.checkbox(
        "LSTM Modelini Kullan", 
        value=pytorch_available,
        disabled=not pytorch_available,
        help="LSTM modeli için PyTorch gereklidir" if not pytorch_available else None
    )
    
    if not pytorch_available and not use_lstm:
        st.warning("⚠️ PyTorch yüklü değil. LSTM modeli kullanılamaz.")
    
    confidence_threshold = st.slider(
        "Güven Eşiği", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6,
        help="Mevsimsel sınıflandırma için güven eşiği"
    )
    
    detect_breakpoints = st.checkbox("Kırılma Noktalarını Tespit Et", value=True)
    
    st.markdown("---")
    st.markdown("### Veri Sütunları")
    date_col = st.text_input("Tarih Sütunu Adı", "")
    value_col = st.text_input("Değer Sütunu Adı", "")
    date_format = st.text_input("Tarih Formatı (opsiyonel)", placeholder="Örn: %Y-%m-%d")
    
    st.markdown("---")
    st.markdown("### Hakkında")
    st.info("Bu uygulama, zaman serilerinde mevsimsellik tespiti yapar.")

# CSV Yükleme Alanı
st.subheader("📂 CSV Dosyası Yükle")
st.markdown("Analiz etmek istediğiniz zaman serisi verilerini içeren bir CSV dosyası yükleyin.")

uploaded_file = st.file_uploader("CSV dosyanızı seçin", type="csv")

# Analiz Fonksiyonu
def analyze_csv(file, date_column, value_column, date_format, use_xgboost, use_lstm, confidence, detect_bp):
    try:
        # CSV'yi yükle
        df = pd.read_csv(file)
        
        # Sütun adlarını belirle
        if not date_column:
            date_column = df.columns[0]  # İlk sütun
            st.info(f"Tarih sütunu: '{date_column}' (otomatik seçildi)")
        
        if not value_column:
            value_column = df.columns[1]  # İkinci sütun
            st.info(f"Değer sütunu: '{value_column}' (otomatik seçildi)")
            
        # Tarih sütununu dönüştür
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])
            
        # Zaman serisi oluştur
        time_series = pd.Series(df[value_column].values, index=df[date_column])
        
        # Zaman serisini sırala
        time_series = time_series.sort_index()
        
        # Mevsimsellik tespit edici oluştur
        detector = SeasonalityDetector(
            use_xgboost=use_xgboost,
            use_lstm=use_lstm,
            confidence_threshold=confidence
        )
        
        # Zaman serisini analiz et
        results = detector.analyze_time_series(time_series, detect_breakpoints=detect_bp)
        
        return time_series, results
    
    except Exception as e:
        st.error(f"Hata: {str(e)}")
        return None, None

# CSV yüklendiyse analiz yap
if uploaded_file is not None:
    # Dosya bilgilerini göster
    file_size = len(uploaded_file.getvalue()) / 1024  # KB cinsinden
    st.write(f"Dosya boyutu: {file_size:.2f} KB")
    
    # Veriyi göster
    df_preview = pd.read_csv(uploaded_file)
    st.write("Veri Önizleme:")
    st.dataframe(df_preview.head())
    
    # En az bir model seçilmiş mi kontrol et
    if not use_xgboost and not use_lstm:
        st.warning("⚠️ Lütfen en az bir model seçin (XGBoost veya LSTM).")
    
    # Analiz butonu
    if st.button("🔍 Analiz Et"):
        if not use_xgboost and not use_lstm:
            st.error("❌ Analiz için en az bir model seçmeniz gerekiyor!")
        else:
            with st.spinner("Analiz yapılıyor..."):
                # Dosyayı başa sar (çünkü önceki read_csv ile dosya sonuna gelmiş olabilir)
                uploaded_file.seek(0)
                
                # Analiz et
                time_series, results = analyze_csv(
                    uploaded_file,
                    date_col,
                    value_col,
                    date_format,
                    use_xgboost,
                    use_lstm,
                    confidence_threshold,
                    detect_breakpoints
                )
                
                if time_series is not None and results is not None:
                    # Sonuçları sekmeler halinde göster
                    tabs = st.tabs(["Analiz Sonuçları", "Grafik", "Kırılma Noktaları", "Özet"])
                    
                    # Tab 1: Analiz Sonuçları
                    with tabs[0]:
                        st.subheader("📊 Analiz Sonuçları")
                        
                        # Mevsimsellik sonucu
                        is_seasonal = results['is_seasonal']
                        if is_seasonal:
                            st.success("✅ Bu zaman serisi MEVSİMSELDİR.")
                        else:
                            st.error("❌ Bu zaman serisi MEVSİMSEL DEĞİLDİR.")
                        
                        # Güven skoru
                        st.metric("Güven Skoru", f"{results['confidence']:.4f}")
                        
                        # Kullanılan model
                        st.write(f"Kullanılan Model: **{results['method'].upper()}**")
                        
                        # Dönem bilgisi
                        if results['period'] is not None:
                            st.write(f"Tespit Edilen Periyot: **{results['period']}** gün")
                            st.write(f"Mevsimsel Güç: **{results['seasonal_strength']:.4f}**")
                    
                    # Tab 2: Grafik
                    with tabs[1]:
                        st.subheader("📈 Zaman Serisi Grafiği")
                        
                        # Grafik oluştur
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(time_series.index, time_series.values)
                        
                        # Kırılma noktalarını ekle
                        if results['breakpoints']:
                            for bp in results['breakpoints']:
                                position = bp['position']
                                if position < len(time_series):
                                    date = time_series.index[position]
                                    ax.axvline(x=date, color='r', linestyle='--', alpha=0.7)
                                    
                                    # Kırılma noktası etiketi ekle
                                    label_text = f"Kırılma\n{date.strftime('%Y-%m-%d')}"
                                    if 'old_period' in bp and 'new_period' in bp:
                                        label_text += f"\n{bp['old_period']:.1f} → {bp['new_period']:.1f} gün"
                                    
                                    y_pos = time_series.max() * 0.9
                                    ax.annotate(label_text, xy=(date, y_pos), 
                                                xytext=(date, y_pos),
                                                ha='center', va='bottom',
                                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
                        
                        plt.title("Zaman Serisi" + (" (Mevsimsel)" if results['is_seasonal'] else " (Mevsimsel Değil)"))
                        plt.tight_layout()
                        
                        # Streamlit'te göster
                        st.pyplot(fig)
                        
                        # Sonuçları indirme butonu
                        results_json = pd.Series(results).to_json()
                        st.download_button(
                            label="📥 Sonuçları İndir (JSON)",
                            data=results_json,
                            file_name="seasonality_results.json",
                            mime="application/json"
                        )
                    
                    # Tab 3: Kırılma Noktaları
                    with tabs[2]:
                        st.subheader("⚡ Kırılma Noktaları")
                        
                        if results['breakpoints']:
                            # Kırılma noktalarını tablo olarak göster
                            bp_data = []
                            for i, bp in enumerate(results['breakpoints']):
                                row = {
                                    "No": i+1,
                                    "Pozisyon": bp['position']
                                }
                                
                                if 'date' in bp:
                                    row["Tarih"] = bp['date'].strftime('%Y-%m-%d') if isinstance(bp['date'], (pd.Timestamp, datetime)) else bp['date']
                                
                                if 'old_period' in bp and 'new_period' in bp:
                                    row["Eski Periyot (gün)"] = f"{bp['old_period']:.1f}"
                                    row["Yeni Periyot (gün)"] = f"{bp['new_period']:.1f}"
                                
                                if 'old_strength' in bp and 'new_strength' in bp:
                                    row["Eski Güç"] = f"{bp['old_strength']:.2f}"
                                    row["Yeni Güç"] = f"{bp['new_strength']:.2f}"
                                    
                                bp_data.append(row)
                            
                            # Tabloyu göster
                            st.dataframe(pd.DataFrame(bp_data).set_index("No"))
                            
                            # Her kırılma noktası için detaylı bilgi
                            for i, bp in enumerate(results['breakpoints']):
                                with st.expander(f"Kırılma Noktası {i+1} Detayları"):
                                    col_a, col_b = st.columns(2)
                                    
                                    with col_a:
                                        st.write(f"**Pozisyon:** {bp['position']}")
                                        if 'date' in bp:
                                            date_str = bp['date'].strftime('%Y-%m-%d') if isinstance(bp['date'], (pd.Timestamp, datetime)) else bp['date']
                                            st.write(f"**Tarih:** {date_str}")
                                    
                                    with col_b:
                                        if 'old_period' in bp and 'new_period' in bp:
                                            st.write(f"**Periyot Değişimi:** {bp['old_period']:.1f} → {bp['new_period']:.1f} gün")
                                        
                                        if 'old_strength' in bp and 'new_strength' in bp:
                                            st.write(f"**Güç Değişimi:** {bp['old_strength']:.2f} → {bp['new_strength']:.2f}")
                                    
                                    # Kırılma noktasını gösteren mini grafik
                                    if position < len(time_series):
                                        # Kırılma öncesi ve sonrası veriyi al
                                        window = min(100, len(time_series) // 3)
                                        start_idx = max(0, position - window)
                                        end_idx = min(len(time_series), position + window)
                                        
                                        segment = time_series.iloc[start_idx:end_idx]
                                        
                                        fig, ax = plt.subplots(figsize=(8, 3))
                                        ax.plot(segment.index, segment.values)
                                        
                                        # Kırılma noktasını işaretle
                                        date = time_series.index[position]
                                        ax.axvline(x=date, color='r', linestyle='--', alpha=0.7)
                                        ax.set_title(f"Kırılma Noktası: {date_str}")
                                        plt.tight_layout()
                                        
                                        st.pyplot(fig)
                        else:
                            st.info("Bu zaman serisinde kırılma noktası tespit edilmedi.")
                    
                    # Tab 4: Özet
                    with tabs[3]:
                        st.subheader("📋 Analiz Özeti")
                        
                        # Ana özet bilgileri
                        summary = f"""
                        ### Genel Sonuçlar
                        **Zaman Serisi**: {'Mevsimsel' if is_seasonal else 'Mevsimsel Değil'}  
                        **Güven Skoru**: {results['confidence']:.4f}  
                        **Kullanılan Model**: {results['method'].upper()}  
                        """
                        
                        if results['period'] is not None:
                            summary += f"**Tespit Edilen Periyot**: {results['period']} gün  \n"
                            summary += f"**Mevsimsel Güç**: {results['seasonal_strength']:.4f}  \n"
                        
                        if results['breakpoints']:
                            summary += f"\n### Kırılma Noktaları\n"
                            summary += f"**Tespit Edilen Kırılma Noktası Sayısı**: {len(results['breakpoints'])}  \n\n"
                            
                            for i, bp in enumerate(results['breakpoints']):
                                date_str = "N/A"
                                if 'date' in bp:
                                    date_str = bp['date'].strftime('%Y-%m-%d') if isinstance(bp['date'], (pd.Timestamp, datetime)) else bp['date']
                                
                                summary += f"**Kırılma {i+1}**: Pozisyon={bp['position']}, Tarih={date_str}  \n"
                                
                                if 'old_period' in bp and 'new_period' in bp:
                                    summary += f"Periyot Değişimi: {bp['old_period']:.1f} → {bp['new_period']:.1f} gün  \n"
                        
                        st.markdown(summary)

else:
    # Örnek CSV
    st.info("Henüz bir CSV dosyası yüklemediniz. Lütfen analiz için bir CSV dosyası yükleyin.")
    
    with st.expander("CSV Formatı Hakkında"):
        st.markdown("""
        ### CSV Formatı
        
        Yüklediğiniz CSV dosyası şu formatta olmalıdır:
        
        ```
        tarih,deger
        2020-01-01,10.5
        2020-01-02,11.2
        2020-01-03,9.8
        ...
        ```
        
        - İlk sütun tarih bilgisi içermelidir.
        - İkinci sütun analiz edilecek sayısal değerleri içermelidir.
        - CSV dosyasında başlık satırı bulunmalıdır.
        - Tarih sütunu standart bir formatta olmalıdır (örn. YYYY-MM-DD).
        """)

# Sayfa altı
st.markdown("---")
st.markdown("📊 Zaman Serisi Mevsimsellik Analizi | v1.0")