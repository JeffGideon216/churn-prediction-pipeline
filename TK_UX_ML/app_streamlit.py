import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ============================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="🔮 Sistema de Predição de Churn",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 2. CARREGAR OS ARTEFATOS
# ============================================
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo e scaler com cache"""
    try:
        pipeline = joblib.load('pipeline_churn_final.pkl')
        scaler = joblib.load('scaler_churn_final.pkl')
        return pipeline, scaler
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None, None

pipeline, scaler = carregar_modelo()

if pipeline is None:
    st.stop()

modelo = pipeline['modelo']
features_esperadas = pipeline['features']
threshold = pipeline['threshold']
modelo_nome = pipeline['modelo_nome']

# ============================================
# 3. FUNÇÕES DE PREDIÇÃO
# ============================================
def prever_cliente(dados):
    """Faz a predição para um único cliente"""
    dados_norm = scaler.transform([dados])
    
    if hasattr(modelo, 'predict_proba'):
        prob = modelo.predict_proba(dados_norm)[0, 1]
    else:
        prob = modelo.predict(dados_norm, verbose=0).flatten()[0]
    
    status = "🔴 RISCO DE CHURN" if prob >= threshold else "🟢 CLIENTE ATIVO"
    recomendacao = "📞 Ação Imediata" if prob >= threshold else "📧 Manter Comunicação"
    
    return prob, status, recomendacao

def processar_planilha(df):
    """Processa uma planilha completa"""
    X_novos = df[features_esperadas].copy()
    X_novos_norm = scaler.transform(X_novos)
    
    if hasattr(modelo, 'predict_proba'):
        probs = modelo.predict_proba(X_novos_norm)[:, 1]
    else:
        probs = modelo.predict(X_novos_norm, verbose=0).flatten()
    
    df['Prob_Churn_%'] = (probs * 100).round(2)
    df['Status'] = ['🔴 RISCO DE CHURN' if p >= threshold else '🟢 CLIENTE ATIVO' 
                    for p in probs]
    df['Recomendacao'] = ['📞 Ação Imediata' if p >= threshold else '📧 Manter Comunicação'
                          for p in probs]
    df['Nível_Risco'] = ['Alto' if p >= 0.7 else 'Médio' if p >= 0.5 else 'Baixo' 
                         for p in probs]
    
    return df, probs

# ============================================
# 4. INTERFACE PRINCIPAL
# ============================================
st.title("🔮 Sistema de Predição de Churn")
st.markdown(f"**Modelo:** {modelo_nome} | **Threshold:** {threshold:.4f}")

# Sidebar com informações
with st.sidebar:
    st.header("📊 Informações")
    st.markdown(f"""
    - **Modelo:** {modelo_nome}
    - **Features:** {len(features_esperadas)}
    - **Threshold:** {threshold:.4f}
    - **Status:** 🟢 Online
    """)
    
    st.divider()
    
    st.header("📂 Selecionar Planilha")
    uploaded_file = st.file_uploader(
        "Faça upload da planilha com dados dos clientes",
        type=['xlsx', 'xls', 'csv']
    )
    
    st.divider()
    
    st.header("🔍 Testar Cliente Único")
    with st.expander("Inserir dados manualmente"):
        idade = st.number_input("Idade", 18, 120, 35)
        total_itens = st.number_input("Total Itens Comprados", 0, 1000, 500)
        total_devolucoes = st.number_input("Total Itens Devolvidos", 0, 100, 0)
        total_pedidos = st.number_input("Total Pedidos", 0, 200, 12)
        valor_gasto = st.number_input("Valor Total Gasto (R$)", 0.0, 100000.0, 2400.0)
        ticket_medio = st.number_input("Ticket Médio (R$)", 0.0, 5000.0, 200.0)
        
        if st.button("🔮 Prever Cliente", type="primary"):
            dados = [idade, total_itens, total_devolucoes, total_pedidos, valor_gasto, ticket_medio]
            prob, status, recomendacao = prever_cliente(dados)
            
            st.markdown("### 📊 Resultado")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilidade de Churn", f"{prob*100:.1f}%")
            with col2:
                st.metric("Status", status)
            st.info(f"📌 {recomendacao}")

# ============================================
# 5. PROCESSAR UPLOAD
# ============================================
if uploaded_file is not None:
    try:
        # Ler o arquivo
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validar colunas
        colunas_faltando = [col for col in features_esperadas if col not in df.columns]
        if colunas_faltando:
            st.error(f"❌ Colunas faltando: {colunas_faltando}")
            st.info(f"Colunas esperadas: {features_esperadas}")
        else:
            # Processar
            with st.spinner("⏳ Processando predições..."):
                df_resultado, probs = processar_planilha(df)
            
            # Estatísticas
            total = len(df_resultado)
            risco = len(df_resultado[df_resultado['Status'] == '🔴 RISCO DE CHURN'])
            ativo = total - risco
            
            # Métricas
            st.subheader("📊 Estatísticas da Análise")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Clientes", total)
            with col2:
                st.metric("🔴 Em Risco", risco, delta=f"{risco/total*100:.1f}%")
            with col3:
                st.metric("🟢 Ativos", ativo, delta=f"{ativo/total*100:.1f}%", delta_color="inverse")
            with col4:
                st.metric("Taxa de Risco", f"{risco/total*100:.1f}%")
            
            # Gráficos
            st.subheader("📈 Visualizações")
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição de Risco
                fig = px.pie(
                    df_resultado, 
                    names='Status',
                    title='Distribuição de Risco',
                    color='Status',
                    color_discrete_map={'🔴 RISCO DE CHURN': '#ef4444', '🟢 CLIENTE ATIVO': '#22c55e'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribuição por Nível de Risco
                risco_counts = df_resultado['Nível_Risco'].value_counts().reset_index()
                risco_counts.columns = ['Nível', 'Quantidade']
                cores = {'Alto': '#ef4444', 'Médio': '#f59e0b', 'Baixo': '#22c55e'}
                fig = px.bar(
                    risco_counts,
                    x='Nível',
                    y='Quantidade',
                    title='Distribuição por Nível de Risco',
                    color='Nível',
                    color_discrete_map=cores
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de Resultados
            st.subheader("📋 Detalhamento dos Clientes")
            
            # Filtros
            col1, col2, col3 = st.columns(3)
            with col1:
                filtro_status = st.selectbox("Filtrar por Status", ["Todos", "🔴 RISCO DE CHURN", "🟢 CLIENTE ATIVO"])
            with col2:
                filtro_risco = st.selectbox("Filtrar por Nível", ["Todos", "Alto", "Médio", "Baixo"])
            with col3:
                ordenar_por = st.selectbox("Ordenar por", ["Prob. Churn (Maior)", "Prob. Churn (Menor)", "Idade"])
            
            # Aplicar filtros
            df_filtrado = df_resultado.copy()
            if filtro_status != "Todos":
                df_filtrado = df_filtrado[df_filtrado['Status'] == filtro_status]
            if filtro_risco != "Todos":
                df_filtrado = df_filtrado[df_filtrado['Nível_Risco'] == filtro_risco]
            
            # Ordenar
            if ordenar_por == "Prob. Churn (Maior)":
                df_filtrado = df_filtrado.sort_values('Prob_Churn_%', ascending=False)
            elif ordenar_por == "Prob. Churn (Menor)":
                df_filtrado = df_filtrado.sort_values('Prob_Churn_%', ascending=True)
            elif ordenar_por == "Idade":
                df_filtrado = df_filtrado.sort_values('Idade')
            
            # Exibir tabela
            colunas_exibir = ['Idade', 'TotalItensComprados', 'TotalItensDevolvidos', 
                            'TotalPedidos', 'ValorTotalGasto', 'TicketMedio', 
                            'Prob_Churn_%', 'Status', 'Recomendacao', 'Nível_Risco']
            
            st.dataframe(
                df_filtrado[colunas_exibir],
                use_container_width=True,
                height=400,
                column_config={
                    "Prob_Churn_%": st.column_config.NumberColumn(
                        "Prob. Churn",
                        format="%.1f%%"
                    ),
                    "ValorTotalGasto": st.column_config.NumberColumn(
                        "Gasto Total",
                        format="R$ %.2f"
                    ),
                    "TicketMedio": st.column_config.NumberColumn(
                        "Ticket Médio",
                        format="R$ %.2f"
                    ),
                    "Status": st.column_config.Column(
                        "Status",
                        width="medium"
                    ),
                    "Recomendacao": st.column_config.Column(
                        "Recomendação",
                        width="medium"
                    )
                }
            )
            
            # Botão de Download
            csv = df_resultado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Resultados (CSV)",
                data=csv,
                file_name=f"resultados_churn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
            
    except Exception as e:
        st.error(f"❌ Erro ao processar: {str(e)}")
        st.code(str(e))

else:
    # Tela inicial
    st.info("👈 Faça upload de uma planilha no painel lateral para começar")
    
    # Exemplo de como deve ser a planilha
    with st.expander("📋 Exemplo de Planilha"):
        dados_exemplo = {
            'Idade': [35, 70, 45],
            'TotalItensComprados': [500, 20, 150],
            'TotalItensDevolvidos': [0, 10, 2],
            'TotalPedidos': [12, 2, 5],
            'ValorTotalGasto': [2400.0, 300.0, 750.0],
            'TicketMedio': [200.0, 150.0, 150.0]
        }
        df_exemplo = pd.DataFrame(dados_exemplo)
        st.dataframe(df_exemplo, use_container_width=True)
        st.caption("A planilha deve conter estas colunas exatamente com estes nomes")