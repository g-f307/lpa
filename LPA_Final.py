import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import altair as alt 
import datetime

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Análise Climática Avançada")

API_KEY = "f323af5a10043e6368d0d173fc7cae17" 
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

@st.cache_data 
def buscar_dados_climaticos(cidade):
    """
    Função para buscar dados climáticos de uma cidade usando a API do OpenWeatherMap.
    Retorna um DataFrame do Pandas com os dados processados ou None em caso de erro.
    """
    params = {
        "q": cidade,
        "appid": API_KEY,
        "units": "metric", 
        "lang": "pt_br"    
    }
    
    try:
        with st.spinner(f"Buscando dados climáticos para {cidade}..."):
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status() 
            data = response.json()

        if data and data.get("list"):
            df = pd.DataFrame(data["list"])
            df["dt_txt"] = pd.to_datetime(df["dt_txt"]) 

            df_main = pd.json_normalize(df['main'])
            
            df_wind = pd.json_normalize(df['wind'])
            df_wind.rename(columns={'speed': 'wind_speed'}, inplace=True) 
            
            df['description'] = df['weather'].apply(lambda x: x[0]['description'] if x and len(x) > 0 else 'N/A')

            df_processed = pd.concat([df['dt_txt'], df_main, df_wind, df['description']], axis=1)

            df_processed = df_processed[['dt_txt', 'temp', 'humidity', 'wind_speed', 'description']].copy()

            df_processed['hour'] = df_processed['dt_txt'].dt.hour
            df_processed['day_of_week'] = df_processed['dt_txt'].dt.dayofweek 
            df_processed['day_of_year'] = df_processed['dt_txt'].dt.dayofyear
            df_processed['month'] = df_processed['dt_txt'].dt.month
            df_processed['time_since_start_hours'] = (df_processed['dt_txt'] - df_processed['dt_txt'].min()).dt.total_seconds() / 3600
            
            return df_processed
        else:
            st.warning(f"Nenhum dado encontrado para a cidade: {cidade}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Não foi possível conectar à API do OpenWeatherMap. Verifique sua conexão ou a chave da API. Erro: {e}")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao buscar dados para {cidade}: {e}")
        return None

def prever_temperatura(df_dados_completos):
    """
    Treina um modelo de regressão linear para prever a temperatura
    e retorna o valor previsto, o timestamp da previsão e métricas de avaliação.
    """
    if df_dados_completos is None or df_dados_completos.empty:
        return None, None, "Dados insuficientes para previsão."

    features = ['hour', 'day_of_week', 'day_of_year', 'month', 'time_since_start_hours', 'temp', 'humidity', 'wind_speed']
    
    current_features = [f for f in features if f in df_dados_completos.columns]
            
    if not current_features:
        return None, None, "Nenhuma feature válida encontrada no DataFrame para previsão."

    X = df_dados_completos[current_features].iloc[:-1] 
    y = df_dados_completos['temp'].iloc[1:]    

    if len(X) != len(y):
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]

    if X.empty:
        return None, None, "Não há dados suficientes para treinar o modelo de previsão."

    model = LinearRegression()
    model.fit(X, y)

    last_dt_txt = df_dados_completos['dt_txt'].iloc[-1]
    next_dt_txt = last_dt_txt + pd.Timedelta(hours=3) 
    next_point_features_df = pd.DataFrame([{
        'hour': next_dt_txt.hour,
        'day_of_week': next_dt_txt.dayofweek,
        'day_of_year': next_dt_txt.dayofyear,
        'month': next_dt_txt.month,
        'time_since_start_hours': (next_dt_txt - df_dados_completos['dt_txt'].min()).total_seconds() / 3600,
        'temp': df_dados_completos['temp'].iloc[-1], 
        'humidity': df_dados_completos['humidity'].iloc[-1],
        'wind_speed': df_dados_completos['wind_speed'].iloc[-1]
    }])
    
    next_point_features_df = next_point_features_df[current_features]

    predicted_temp = model.predict(next_point_features_df)[0]
    
    y_pred_train = model.predict(X)
    mae = mean_absolute_error(y, y_pred_train)
    mse = mean_squared_error(y, y_pred_train)

    return predicted_temp, next_dt_txt, (
        f"Previsão para {next_dt_txt.strftime('%d/%m %H:%M')}: **{predicted_temp:.2f}°C**\n\n"
        f"Métricas (nos dados de previsão existentes):\nMAE: {mae:.2f}\nMSE: {mse:.2f}"
    )

# --- Layout da Interface Streamlit ---
st.title("🌎 Análise Climática Avançada")
st.markdown("Use esta aplicação para visualizar dados climáticos e estatísticas para diversas cidades.")
with st.sidebar:
    st.header("Configurações")
    cidade_principal = st.text_input("Digite o nome da cidade principal:", "Manaus", key="cidade_principal")
    st.subheader("Comparar com Outra Cidade (Opcional)")
    cidade_comparacao = st.text_input("Digite o nome da segunda cidade:", "", key="cidade_comparacao")
    periodos = ["Próximas 24h", "Próximos 3 dias", "Próximos 5 dias"]
    if 'periodo_selecionado' not in st.session_state:
        st.session_state.periodo_selecionado = periodos[2]

    st.session_state.periodo_selecionado = st.selectbox(
        "Selecione o Período:", 
        periodos, 
        index=periodos.index(st.session_state.periodo_selecionado), 
        key='periodo_dropdown' 
    )

    tipo_grafico_map = {
        "Temperatura": "temp",
        "Umidade": "humidity",
        "Velocidade do Vento": "wind_speed"
    }
    tipos_grafico_display = list(tipo_grafico_map.keys()) 

    if 'tipo_grafico_selecionado_display' not in st.session_state:
        st.session_state.tipo_grafico_selecionado_display = tipos_grafico_display[0] 

    st.session_state.tipo_grafico_selecionado_display = st.selectbox(
        "Selecione o Tipo de Gráfico:", 
        tipos_grafico_display, 
        index=tipos_grafico_display.index(st.session_state.tipo_grafico_selecionado_display),
        key='tipo_grafico_dropdown' 
    )
    coluna_selecionada_df = tipo_grafico_map[st.session_state.tipo_grafico_selecionado_display]


# --- Área Principal de Conteúdo ---
df_dados_principal = None
if cidade_principal:
    df_dados_principal = buscar_dados_climaticos(cidade_principal)
df_dados_comparacao = None
if cidade_comparacao:
    df_dados_comparacao = buscar_dados_climaticos(cidade_comparacao)
    
st.subheader("📈 Visualização do Clima")
df_para_grafico = pd.DataFrame()
previsao_principal = None
dt_previsao_principal = None
mensagem_previsao_principal = "Carregue os dados climáticos para ver a previsão."
if df_dados_principal is not None and not df_dados_principal.empty:
    df_filtrado_principal = df_dados_principal.copy()
    agora = pd.Timestamp.now() 
    
    if st.session_state.periodo_selecionado == "Próximas 24h":
        df_filtrado_principal = df_filtrado_principal[
            (df_filtrado_principal['dt_txt'] >= agora) & 
            (df_filtrado_principal['dt_txt'] <= agora + pd.Timedelta(days=1))
        ]
    elif st.session_state.periodo_selecionado == "Próximos 3 dias":
        df_filtrado_principal = df_filtrado_principal[
            (df_filtrado_principal['dt_txt'] >= agora) & 
            (df_filtrado_principal['dt_txt'] <= agora + pd.Timedelta(days=3))
        ]
    elif st.session_state.periodo_selecionado == "Próximos 5 dias":
        df_filtrado_principal = df_filtrado_principal[df_filtrado_principal['dt_txt'] >= agora]

    if df_filtrado_principal.empty:
        st.warning(f"Não há dados disponíveis para {cidade_principal} no período '{st.session_state.periodo_selecionado}'.")
    
    df_plot_principal = df_filtrado_principal[['dt_txt', coluna_selecionada_df]].copy()
    df_plot_principal.columns = ['Data e Hora', 'Valor']
    df_plot_principal['Cidade'] = cidade_principal
    df_plot_principal['Tipo'] = 'Observado' 

    previsao_principal, dt_previsao_principal, mensagem_previsao_principal = prever_temperatura(df_dados_principal)
    if previsao_principal is not None and dt_previsao_principal is not None:
        df_previsao_plot_principal = pd.DataFrame({
            'Data e Hora': [dt_previsao_principal],
            'Valor': [previsao_principal],
            'Cidade': cidade_principal,
            'Tipo': ['Previsão'] 
        })
        df_plot_principal = pd.concat([df_plot_principal, df_previsao_plot_principal], ignore_index=True)


    df_para_grafico = df_plot_principal
    if df_dados_comparacao is not None and not df_dados_comparacao.empty:
        df_filtrado_comparacao = df_dados_comparacao.copy()
        if st.session_state.periodo_selecionado == "Próximas 24h":
            df_filtrado_comparacao = df_filtrado_comparacao[
                (df_filtrado_comparacao['dt_txt'] >= agora) & 
                (df_filtrado_comparacao['dt_txt'] <= agora + pd.Timedelta(days=1))
            ]
        elif st.session_state.periodo_selecionado == "Próximos 3 dias":
            df_filtrado_comparacao = df_filtrado_comparacao[
                (df_filtrado_comparacao['dt_txt'] >= agora) & 
                (df_filtrado_comparacao['dt_txt'] <= agora + pd.Timedelta(days=3))
            ]
        elif st.session_state.periodo_selecionado == "Próximos 5 dias":
            df_filtrado_comparacao = df_filtrado_comparacao[df_filtrado_comparacao['dt_txt'] >= agora]

        if df_filtrado_comparacao.empty:
            st.warning(f"Não há dados disponíveis para {cidade_comparacao} no período '{st.session_state.periodo_selecionado}'.")

        df_plot_comparacao = df_filtrado_comparacao[['dt_txt', coluna_selecionada_df]].copy()
        df_plot_comparacao.columns = ['Data e Hora', 'Valor']
        df_plot_comparacao['Cidade'] = cidade_comparacao
        df_plot_comparacao['Tipo'] = 'Observado' 

        df_para_grafico = pd.concat([df_para_grafico, df_plot_comparacao], ignore_index=True)

if not df_para_grafico.empty:
    y_axis_title = f'{st.session_state.tipo_grafico_selecionado_display.capitalize()} ('
    if st.session_state.tipo_grafico_selecionado_display == "Temperatura":
        y_axis_title += "°C)"
    elif st.session_state.tipo_grafico_selecionado_display == "Umidade":
        y_axis_title += "%)"
    elif st.session_state.tipo_grafico_selecionado_display == "Velocidade do Vento":
        y_axis_title += "m/s)"
    
    chart = alt.Chart(df_para_grafico).mark_line(point=True).encode(
        x=alt.X('Data e Hora', axis=alt.Axis(title='Data e Hora')),
        y=alt.Y('Valor', axis=alt.Axis(title=y_axis_title)),
        color=alt.Color('Cidade', title='Cidade'), 
        strokeDash=alt.StrokeDash('Tipo', legend=alt.Legend(title='Tipo de Dado')), 
        tooltip=['Data e Hora', 'Valor', 'Cidade', 'Tipo'] 
    ).properties(
        title=f'{st.session_state.tipo_grafico_selecionado_display.capitalize()} ao Longo do Tempo'
    ).interactive() 

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Digite o nome de uma cidade para começar a visualizar os dados climáticos.")


# --- Seção de Estatísticas e Previsão Numérica (Colunas) ---
col_stats, col_previsao = st.columns(2)

with col_stats:
    st.subheader("📊 Estatísticas Resumidas")
    if df_dados_principal is not None and not df_dados_principal.empty:
        df_filtrado_stats = df_dados_principal.copy() 
        agora_stats = pd.Timestamp.now() 
        if st.session_state.periodo_selecionado == "Próximas 24h":
            df_filtrado_stats = df_filtrado_stats[
                (df_filtrado_stats['dt_txt'] >= agora_stats) & 
                (df_filtrado_stats['dt_txt'] <= agora_stats + pd.Timedelta(days=1))
            ]
        elif st.session_state.periodo_selecionado == "Próximos 3 dias":
            df_filtrado_stats = df_filtrado_stats[
                (df_filtrado_stats['dt_txt'] >= agora_stats) & 
                (df_filtrado_stats['dt_txt'] <= agora_stats + pd.Timedelta(days=3))
            ]
        elif st.session_state.periodo_selecionado == "Próximos 5 dias":
            df_filtrado_stats = df_filtrado_stats[df_filtrado_stats['dt_txt'] >= agora_stats]

        if not df_filtrado_stats.empty:
            temp_max = df_filtrado_stats['temp'].max() if 'temp' in df_filtrado_stats.columns else "N/A"
            temp_min = df_filtrado_stats['temp'].min() if 'temp' in df_filtrado_stats.columns else "N/A"
            temp_media = df_filtrado_stats['temp'].mean() if 'temp' in df_filtrado_stats.columns else "N/A"
            umidade_media = df_filtrado_stats['humidity'].mean() if 'humidity' in df_filtrado_stats.columns else "N/A"
            vento_medio = df_filtrado_stats['wind_speed'].mean() if 'wind_speed' in df_filtrado_stats.columns else "N/A"

            st.metric("Período Analisado", st.session_state.periodo_selecionado)
            st.metric("Temperatura Média (°C)", f"{temp_media:.2f}" if isinstance(temp_media, (int, float)) else temp_media)
            st.metric("Temperatura Máxima (°C)", f"{temp_max:.2f}" if isinstance(temp_max, (int, float)) else temp_max)
            st.metric("Temperatura Mínima (°C)", f"{temp_min:.2f}" if isinstance(temp_min, (int, float)) else temp_min)
            st.metric("Umidade Média (%)", f"{umidade_media:.2f}" if isinstance(umidade_media, (int, float)) else umidade_media)
            st.metric("Velocidade Média do Vento (m/s)", f"{vento_medio:.2f}" if isinstance(vento_medio, (int, float)) else vento_medio)
        else:
            st.warning(f"Não há dados para estatísticas de {cidade_principal} no período '{st.session_state.periodo_selecionado}'.")
    else:
        st.info("Carregue os dados climáticos para ver as estatísticas.")

with col_previsao:
    st.subheader(f"🔮 Previsão de Temperatura para {cidade_principal}")
    if df_dados_principal is not None and not df_dados_principal.empty:
        previsao_val, dt_previsao_val, mensagem_previsao_val = prever_temperatura(df_dados_principal)
        if previsao_val is not None and dt_previsao_val is not None:
            st.write(f"**Temperatura Prevista:** {previsao_val:.2f}°C")
            st.info(mensagem_previsao_val)
        else:
            st.warning(mensagem_previsao_val)
    else:
        st.info("Carregue os dados climáticos para ver a previsão.")

# --- Seção de Dados Brutos ---
st.subheader("Raw Data (Dados Brutos)")
with st.expander("Ver Dados Detalhados"):
    dataframes_para_exibir = {}
    if df_dados_principal is not None and not df_dados_principal.empty:
        dataframes_para_exibir[f"{cidade_principal} ({st.session_state.periodo_selecionado})"] = df_filtrado_principal
    if df_dados_comparacao is not None and not df_dados_comparacao.empty:
        dataframes_para_exibir[f"{cidade_comparacao} ({st.session_state.periodo_selecionado})"] = df_filtrado_comparacao

    if dataframes_para_exibir:
        cidade_para_tabela = st.selectbox(
            "Selecione a cidade para ver os dados detalhados:", 
            list(dataframes_para_exibir.keys()),
            key='select_raw_data_city'
        )
        st.dataframe(dataframes_para_exibir[cidade_para_tabela], use_container_width=True)
    else:
        st.info("Nenhum dado disponível para exibição na tabela.")
