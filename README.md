# 🏢 Churn Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![Tkinter](https://img.shields.io/badge/Tkinter-Desktop-lightblue.svg)](https://docs.python.org/3/library/tkinter.html)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pipeline completo de análise preditiva de churn utilizando dados estruturados. O projeto engloba desde a extração via SQLAlchemy, balanceamento com SMOTE, modelagem avançada (XGBoost, Random Forest e Redes Neurais) até o deploy multiplataforma: uma aplicação web interativa em **Streamlit** e um software desktop nativo em **Tkinter**.

---

## 🎯 Objetivo

Identificar clientes com alto risco de **churn** (cancelamento/evasão) antes que eles parem de comprar, permitindo ações preventivas automatizadas baseadas em insights de Machine Learning, como cálculo de probabilidade e recomendações de ações.

---

## 🖥️ Interfaces de Deploy

Para atender a diferentes perfis de usuários (equipes de negócios na web ou analistas locais), o modelo foi implantado em dois formatos de interface gráfica:

### 1. Dashboard Web (Streamlit)
Ideal para apresentações executivas e monitoramento de métricas em tempo real. Permite o upload de arquivos `.xlsx`, exibe gráficos de distribuição de risco e tabelas interativas de clientes.

<p align="center">
  <img src="ST_UX_PREDICAO.png" alt="Streamlit Dashboard Overview" width="45%"/>
  <img src="ST_UX_PREDICAO 2.png" alt="Streamlit Detalhamento" width="45%"/>
</p>

### 2. Aplicativo Desktop (Tkinter / CustomTkinter)
Uma solução robusta e leve para execução local offline, contando com alternância dinâmica entre temas claro e escuro.

<p align="center">
  <img src="TK_UX_PREVISAO_DARK.png" alt="Tkinter Tema Dark" width="45%"/>
  <img src="TK_UX_PREVISAO_LIGTH.png" alt="Tkinter Tema Light" width="45%"/>
</p>

---

## 🛠️ Tecnologias Utilizadas

### 📦 Engenharia & Pré-processamento
- **SQLAlchemy & pyodbc** – Conexão e extração de dados do SQL Server.
- **Pandas & NumPy** – Manipulação, limpeza e engenharia de recursos.
- **SMOTE (Imbalanced-Learn)** – Balanceamento estatístico da classe minoritária.

### 🤖 Modelagem & Avaliação
- **XGBoost & Random Forest** – Algoritmos de árvore de decisão de alta performance.
- **TensorFlow/Keras** – Redes Neurais Profundas com camadas de Dropout e BatchNorm.
- **Scikit-Learn** – Validação, métricas de classificação (Recall, ROC-AUC) e pipelines.

---

## 📈 Resultados dos Modelos

| Modelo | Balanceamento | Recall (Churn) | ROC-AUC | F1-Score |
|:---|:---|:---|:---|:---|
| **Random Forest** | SMOTE | 76.7% | 0.955 | 0.834 |
| **XGBoost** | SMOTE | 80.7% | **0.963** | **0.851** |
| **Rede Neural (TF)** | class_weight | **83.1%** | 0.956 | 0.828 |

> 🏆 **Destaque:** Embora o XGBoost apresente a melhor consistência geral (ROC-AUC de 0.963), a Rede Neural obteve o maior **Recall (83.1%)**, sendo a mais eficiente em não deixar clientes em risco passarem despercebidos.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python 3.11+
- Banco de dados configurado (ou arquivo de dados correspondente)

### 1. Instalação
```bash
# Clone o repositório
git clone [https://github.com/JeffGideon216/churn-prediction-pipeline.git](https://github.com/JeffGideon216/churn-prediction-pipeline.git)
cd churn-prediction-pipeline

# Instale as dependências
pip install -r requirements.txt