# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import warnings
from datetime import datetime, timedelta

# Configuración básica
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Predicción IPC-LSTM & Analytics", layout="wide")

st.header(":material/graph_3: Panel Predictivo: IPC (Modelo LSTM) & Analytics")

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
        
        if values_clean.notnull().sum() > 24: # Al menos 2 años de datos
            # 1. Interpolar y limpiar la serie temporal
            serie_temp = pd.Series(values_clean).interpolate(method='linear').bfill().ffill()
            
            # 2. CALCULAR LA VARIACIÓN PORCENTUAL MENSUAL (Tasa de inflación)
            variacion_pct = (serie_temp.pct_change() * 100).fillna(0).values
            
            cleaned_data.append({
                'Region': current_region,
                'Item': item_name,
                'Values': variacion_pct # Trabajamos sobre la tasa (%) directamente
            })
    
    return cleaned_data, dates_formatted

def entrenar_y_predecir_lstm(serie, seq_length=24, steps=3):
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
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
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
    region_sel = st.sidebar.selectbox("Región", sorted(list(set([d['Region'] for d in data_list]))))
    
    if st.button("Calcular Proyecciones LSTM"):
        procesar = [d for d in data_list if d['Region'] == region_sel]
        resultados = []
        bar = st.progress(0)
        
        for i, entry in enumerate(procesar):
            preds = entrenar_y_predecir_lstm(entry['Values'])
            if preds is not None:
                resultados.append({
                    "Ítem": entry['Item'],
                    "Último Valor (%)": round(entry['Values'][-1], 2),
                    "Mes 1 (%)": round(preds[0], 2),
                    "Mes 2 (%)": round(preds[1], 2),
                    "Mes 3 (%)": round(preds[2], 2)
                })
            bar.progress((i + 1) / len(procesar))
        st.session_state['res_lstm'] = pd.DataFrame(resultados)

    if 'res_lstm' in st.session_state:
        df_res = st.session_state['res_lstm']
        
        # Tabla formateada visualmente con símbolo %
        st.dataframe(
            df_res.style.format({
                "Último Valor (%)": "{:.2f}%",
                "Mes 1 (%)": "{:.2f}%",
                "Mes 2 (%)": "{:.2f}%",
                "Mes 3 (%)": "{:.2f}%"
            }),
            width='stretch', 
            hide_index=True
        )

        st.divider()
        item_graf = st.selectbox("Seleccione Rubro:", df_res['Ítem'].unique())
        
        try:
            entry = next(d for d in data_list if d['Item'] == item_graf and d['Region'] == region_sel)
            hist_vals = entry['Values'][-24:] # Últimos 2 años en %
            hist_dates = dates_hist[-24:]
            
            row_p = df_res[df_res['Ítem'] == item_graf].iloc[0]
            preds_pct = [row_p['Mes 1 (%)'], row_p['Mes 2 (%)'], row_p['Mes 3 (%)']]
            
            u_fec = datetime.strptime(hist_dates[-1], '%Y-%m')
            fec_fut = [(u_fec + timedelta(days=31*i)).strftime('%Y-%m') for i in range(1, 4)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_dates, y=hist_vals, name='Variación Real (%)', line=dict(color='#2ca02c', width=3), mode='lines+markers'))
            
            x_pred = [hist_dates[-1]] + fec_fut
            y_pred = [hist_vals[-1]] + preds_pct
            fig.add_trace(go.Scatter(x=x_pred, y=y_pred, name='Proyección de Tasa (%)', line=dict(color='#ff7f0e', width=3, dash='dash'), mode='lines+markers'))
            
            fig.update_layout(
                title=f"Tasa de Inflación Mensual (LSTM): {item_graf}", 
                xaxis_title="Meses", 
                yaxis_title="Variación Porcentual (%)", 
                hovermode="x unified", 
                template="plotly_dark", 
                yaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Error al graficar: {e}")
else:
    st.error("Verifique que el archivo esté en data/ipc.xlsx")