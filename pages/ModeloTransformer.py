# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
from datetime import datetime, timedelta

# Configuración y estética
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Predicción IPC-Transformer  & Analytics", layout="wide")

st.header(":material/graph_5: Proyección de Inflación: Modelo Transformer  & Analytics")

@st.cache_data
def cargar_datos_ipc(file_path):
    if not os.path.exists(file_path):
        return None, None

    try:
        # CAMBIO CLAVE: Usamos el nombre exacto de tu solapa
        nombre_hoja = 'Índices IPC Cobertura Nacional'
        raw_df = pd.read_excel(file_path, sheet_name=nombre_hoja, header=None, dtype=str)
    except Exception as e:
        # Si falla, intentamos cargar la primera hoja disponible
        st.warning(f"No se encontró la hoja '{nombre_hoja}', intentando con la primera...")
        raw_df = pd.read_excel(file_path, header=None, dtype=str)

    # Buscamos la fila que contiene las fechas (donde aparece 2016-12-01 o similar)
    mask = raw_df.astype(str).apply(lambda r: r.str.contains('2016-12', na=False).any(), axis=1)
    matching_indices = raw_df.index[mask].tolist()

    if not matching_indices:
        st.error("No se encontró la celda de control '2016-12' en el archivo.")
        return None, None
        
    date_row_idx = matching_indices[0]
    
    # Extraer fechas y formatearlas
    dates_raw = raw_df.iloc[date_row_idx, 1:].replace('nan', np.nan).dropna().values
    dates_formatted = pd.to_datetime(dates_raw).strftime('%Y-%m')
    
    cleaned_data = []
    current_region = "Total nacional"

    # Recorrer los datos desde la fila de fechas hacia abajo
    for idx, row in raw_df.iloc[date_row_idx + 1:].iterrows():
        first_col = str(row[0]).strip()
        
        if pd.isna(first_col) or first_col == "" or "nan" in first_col.lower():
            continue
            
        # Detectar cambios de región (GBA, Pampeana, etc.)
        if any(reg in first_col for reg in ["GBA", "Pampeana", "Noreste", "Noroeste", "Cuyo", "Patagonia"]):
            current_region = first_col
        
        item_name = first_col
        values_raw = row[1:len(dates_raw)+1].values
        values_clean = pd.to_numeric(pd.Series(values_raw).replace(['///', 'nan', ''], np.nan), errors='coerce')
        
        if values_clean.notnull().sum() > 24: # Al menos 2 años de datos
            # 1. Interpolar y limpiar la serie temporal bruta
            serie_temp = pd.Series(values_clean).interpolate(method='linear').bfill().ffill()
            
            # 2. CALCULAR LA VARIACIÓN PORCENTUAL MENSUAL (El paso indispensable)
            variacion_pct = (serie_temp.pct_change() * 100).fillna(0).values
            
            cleaned_data.append({
                'Region': current_region,
                'Item': item_name,
                'Values': variacion_pct # Ahora guardamos el % real
            })
    
    return cleaned_data, dates_formatted


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def entrenar_y_predecir_transformer(serie, seq_length=24, steps=3):
    if len(serie) <= seq_length: return None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(serie.reshape(-1, 1))
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = build_transformer_model((seq_length, 1))
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)
    
    input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    preds_scaled = []
    for _ in range(steps):
        p = model.predict(input_seq, verbose=0)[0][0]
        preds_scaled.append(p)
        new_val = np.array([[[p]]])
        input_seq = np.append(input_seq[:, 1:, :], new_val, axis=1)
    
    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# --- EJECUCIÓN ---
FILE_PATH = "data/ipc.xlsx"
data_list, dates_hist = cargar_datos_ipc(FILE_PATH)

if data_list is not None: 
    st.sidebar.header(":material/settings: Filtros")          
    region_sel = st.sidebar.selectbox("Región", sorted(list(set([d['Region'] for d in data_list]))))
    
    if st.button("Calcular Proyecciones Transformer"):
        procesar = [d for d in data_list if d['Region'] == region_sel]
        resultados = []
        bar = st.progress(0)
        for i, entry in enumerate(procesar):
            preds = entrenar_y_predecir_transformer(entry['Values'])
            if preds is not None:
                resultados.append({
                    "Ítem": entry['Item'],
                    "Último %": round(entry['Values'][-1], 2),
                    "Mes 1": round(preds[0], 2),
                    "Mes 2": round(preds[1], 2),
                    "Mes 3": round(preds[2], 2)
                })
            bar.progress((i + 1) / len(procesar))
        st.session_state['res_trans'] = pd.DataFrame(resultados)

    if 'res_trans' in st.session_state:
        df_res = st.session_state['res_trans']
        
        # Formateamos visualmente la tabla agregando el sufijo '%'
        st.dataframe(
            df_res.style.format({
                "Último %": "{:.2f}%",
                "Mes 1": "{:.2f}%",
                "Mes 2": "{:.2f}%",
                "Mes 3": "{:.2f}%"
            }),
            width='stretch',
            hide_index=True
        )
        
        st.divider()
        st.subheader("Gráfico Dinámico con Zoom Temporal")
        item_graf = st.selectbox("Seleccione Rubro:", df_res['Ítem'].unique())
        
        # --- LÓGICA DE ZOOM ---
        entry_data = next(d for d in data_list if d['Item'] == item_graf and d['Region'] == region_sel)
        full_vals = entry_data['Values']
        full_dates = dates_hist
        
        rango = st.select_slider(
            "Deslice para ajustar el periodo visible:",
            options=list(full_dates),
            value=(full_dates[-24], full_dates[-1])
        )
        
        mask = (full_dates >= rango[0]) & (full_dates <= rango[1])
        vals_zoom = full_vals[mask]
        dates_zoom = full_dates[mask]
        
        row_p = df_res[df_res['Ítem'] == item_graf].iloc[0]
        preds_vals = [row_p['Mes 1'], row_p['Mes 2'], row_p['Mes 3']]
        
        ult_fec_dt = datetime.strptime(full_dates[-1], '%Y-%m')
        fut_dates = [(ult_fec_dt + timedelta(days=31*(i+1))).strftime('%Y-%m') for i in range(3)]
        
        # --- PLOTLY CORREGIDO ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates_zoom, y=vals_zoom, name='Histórico (%)', line=dict(color='#17BECF', width=3), mode='lines+markers'))
        
        x_pred = [dates_zoom[-1]] + fut_dates
        y_pred = [vals_zoom[-1]] + preds_vals
        fig.add_trace(go.Scatter(x=x_pred, y=y_pred, name='Predicción Transformer (%)', 
                                line=dict(color='#FF4B4B', width=4, dash='dash'), mode='lines+markers'))
        
        fig.update_layout(
            title=f"Evolución y Predicción de Tasa Mensual: {item_graf}",
            yaxis_title="Variación Mensual %", 
            template="plotly_dark", 
            hovermode="x unified"
        )
        st.plotly_chart(fig, width='stretch')
else:
    st.error("Verifique que el archivo esté en data/ipc.xlsx")