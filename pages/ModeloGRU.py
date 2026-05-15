# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Predicción IPC-Modelo GRU & Analytics", layout="wide")

st.header(":material/modeling: Análisis Predictivo: IPC-Modelo GRU & Analytics")

st.info("""
**Nota Técnica:** Este sistema utiliza una arquitectura de **Redes Neuronales Recurrentes (GRU)**. 
El modelo se entrena con una secuencia histórica de 24 meses para proyectar los valores futuros.
""")

@st.cache_data
def cargar_datos_ipc(file_path):
    if not os.path.exists(file_path):
        return None, None

    try:
        nombre_hoja = 'Índices IPC Cobertura Nacional'
        raw_df = pd.read_excel(file_path, sheet_name=nombre_hoja, header=None, dtype=str)
    except Exception as e:
        st.warning(f"No se encontró la hoja '{nombre_hoja}', intentando con la primera...")
        raw_df = pd.read_excel(file_path, header=None, dtype=str)

    mask = raw_df.astype(str).apply(lambda r: r.str.contains('2016-12', na=False).any(), axis=1)
    matching_indices = raw_df.index[mask].tolist()

    if not matching_indices:
        st.error("No se encontró la celda de control '2016-12' en el archivo.")
        return None, None
        
    date_row_idx = matching_indices[0]
    dates_raw = raw_df.iloc[date_row_idx, 1:].replace('nan', np.nan).dropna().values
    dates_formatted = pd.to_datetime(dates_raw).strftime('%Y-%m')
    
    cleaned_data = []
    current_region = "Total nacional"

    for idx, row in raw_df.iloc[date_row_idx + 1:].iterrows():
        first_col = str(row[0]).strip()
        
        if pd.isna(first_col) or first_col == "" or "nan" in first_col.lower():
            continue
            
        if any(reg in first_col for reg in ["GBA", "Pampeana", "Noreste", "Noroeste", "Cuyo", "Patagonia"]):
            current_region = first_col
        
        item_name = first_col
        values_raw = row[1:len(dates_raw)+1].values
        values_clean = pd.to_numeric(pd.Series(values_raw).replace(['///', 'nan', ''], np.nan), errors='coerce')
        
        if values_clean.notnull().sum() > 24:
            # 1. Interpolar y limpiar la serie temporal
            serie_temp = pd.Series(values_clean).interpolate(method='linear').bfill().ffill()
            
            # 2. CALCULAR LA VARIACIÓN PORCENTUAL MENSUAL (Tasa de inflación)
            variacion_pct = (serie_temp.pct_change() * 100).fillna(0).values
            
            cleaned_data.append({
                'Region': current_region,
                'Item': item_name,
                'Values': variacion_pct # Guardamos la variación porcentual real
            })
    
    return cleaned_data, dates_formatted

def entrenar_y_predecir_gru(serie, seq_length=24, steps=3):
    if len(serie) <= seq_length: return None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(serie.reshape(-1, 1))

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)

    input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    preds_scaled = []
    for _ in range(steps):
        p = model.predict(input_seq, verbose=0)[0][0]
        preds_scaled.append(p)
        new_val = np.array([[[p]]])
        input_seq = np.append(input_seq[:, 1:, :], new_val, axis=1)
    
    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# --- FLUJO PRINCIPAL ---
FILE_PATH = "data/ipc.xlsx"
data_list, dates_hist = cargar_datos_ipc(FILE_PATH)


if data_list:
    st.sidebar.header(":material/settings: Filtros")  
    region_sel = st.sidebar.selectbox("Región de Análisis", sorted(list(set([d['Region'] for d in data_list]))))
    
    if st.button("Ejecutar Modelo GRU"):
        procesar = [d for d in data_list if d['Region'] == region_sel]
        resultados = []
        progreso = st.progress(0)
        
        for i, entry in enumerate(procesar):
            preds = entrenar_y_predecir_gru(entry['Values'])
            if preds is not None:
                resultados.append({
                    "Ítem": entry['Item'],
                    "Var. M1 (%)": round(preds[0], 2),
                    "Var. M2 (%)": round(preds[1], 2),
                    "Var. M3 (%)": round(preds[2], 2)
                })
            progreso.progress((i + 1) / len(procesar))
        st.session_state['res_gru'] = pd.DataFrame(resultados)

    if 'res_gru' in st.session_state:
        df3 = st.session_state['res_gru']
        st.subheader(f"Predicciones con Modelo GRU - {region_sel}")
        
        # Tabla formateada visualmente con símbolo %
        st.dataframe(
            df3.style.format({
                "Var. M1 (%)": "{:.2f}%",
                "Var. M2 (%)": "{:.2f}%",
                "Var. M3 (%)": "{:.2f}%"
            }),
            width='stretch',
            hide_index=True
        )

        st.divider()
        item_graf = st.selectbox("Seleccione rubro para graficar:", df3['Ítem'].unique())
        
        entry = next(d for d in data_list if d['Item'] == item_graf and d['Region'] == region_sel)
        hist_vals = entry['Values'][-24:] # Últimos 2 años ya calculados como %
        hist_dates = dates_hist[-24:]
        
        row_p = df3[df3['Ítem'] == item_graf].iloc[0]
        preds_pct = [row_p['Var. M1 (%)'], row_p['Var. M2 (%)'], row_p['Var. M3 (%)']]
        
        u_fec = datetime.strptime(hist_dates[-1], '%Y-%m')
        fec_fut = [(u_fec + timedelta(days=31*i)).strftime('%Y-%m') for i in range(1, 4)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_vals, name='Histórico (%)', line=dict(color='#2ca02c', width=3), mode='lines+markers'))
        
        x_pred = [hist_dates[-1]] + fec_fut
        y_pred = [hist_vals[-1]] + preds_pct
        fig.add_trace(go.Scatter(x=x_pred, y=y_pred, name='Proyección (%)', line=dict(color='#ff7f0e', width=3, dash='dash'), mode='lines+markers'))
        
        fig.update_layout(
            title=f"Evolución de la Tasa Mensual (GRU) - {item_graf}", 
            yaxis_title="Variación Mensual %", 
            template="plotly_dark", 
            hovermode="x unified"
        )
        st.plotly_chart(fig, width='stretch')
else:
    st.error("Verifique que el archivo esté en data/ipc.xlsx")