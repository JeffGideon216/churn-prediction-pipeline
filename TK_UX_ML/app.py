import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
import traceback

# ============================================
# 1. CARREGAR OS ARTEFATOS DO MODELO
# ============================================
print("📦 Carregando artefatos do modelo...")

try:
    pipeline = joblib.load('pipeline_churn_final.pkl')
    scaler = joblib.load('scaler_churn_final.pkl')
    modelo = pipeline['modelo']
    features_esperadas = pipeline['features']
    threshold = pipeline['threshold']
    print("✅ Artefatos carregados com sucesso!")
    print(f"   Modelo: {pipeline['modelo_nome']}")
    print(f"   Features: {features_esperadas}")
    print(f"   Threshold: {threshold:.4f}")
except Exception as e:
    print(f"❌ Erro ao carregar artefatos: {e}")
    input("Pressione Enter para sair...")
    exit(1)

# ============================================
# 2. FUNÇÃO PRINCIPAL DE PREDIÇÃO
# ============================================
def processar_planilha(caminho_arquivo):
    """Carrega a planilha, faz as previsões e retorna o DataFrame com os resultados"""
    try:
        df = pd.read_excel(caminho_arquivo)
        
        colunas_faltando = [col for col in features_esperadas if col not in df.columns]
        if colunas_faltando:
            raise ValueError(f"Colunas faltando: {colunas_faltando}")
        
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
        
        return df, probs
        
    except Exception as e:
        raise Exception(f"Erro ao processar: {str(e)}")

# ============================================
# 3. FUNÇÃO PARA CRIAR PLANILHA EXEMPLO
# ============================================
def criar_planilha_exemplo():
    """Cria um arquivo Excel de exemplo para testes (se não existir)"""
    caminho = os.path.join('exemplos', 'clientes_exemplo.xlsx')
    
    if os.path.exists(caminho):
        print(f"✅ Planilha exemplo já existe em: {caminho}")
        return caminho
    
    dados_exemplo = {
        'Idade': [35, 70, 45, 25, 55, 32, 48, 62, 29, 41],
        'TotalItensComprados': [500, 20, 150, 5, 300, 80, 45, 200, 12, 90],
        'TotalItensDevolvidos': [0, 10, 2, 0, 1, 5, 0, 3, 8, 2],
        'TotalPedidos': [12, 2, 5, 1, 8, 4, 3, 10, 2, 6],
        'ValorTotalGasto': [2400.0, 300.0, 750.0, 50.0, 1800.0, 520.0, 280.0, 1500.0, 180.0, 650.0],
        'TicketMedio': [200.0, 150.0, 150.0, 50.0, 225.0, 130.0, 93.3, 150.0, 90.0, 108.3]
    }
    
    df = pd.DataFrame(dados_exemplo)
    os.makedirs('exemplos', exist_ok=True)
    df.to_excel(caminho, index=False)
    print(f"✅ Planilha exemplo criada em: {caminho}")
    return caminho

# ============================================
# 4. INTERFACE GRÁFICA COM TEMA
# ============================================
class ChurnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔮 Sistema de Predição de Churn - Contoso Retail")
        self.root.geometry("1400x850")
        self.root.resizable(True, True)
        
        # Tema atual
        self.tema_atual = "dark"
        
        # Cores do tema Dark (textos mais claros)
        self.cores_dark = {
            'bg_primary': '#0d1117',
            'bg_secondary': '#161b22',
            'bg_tertiary': '#1c2333',
            'bg_card': '#1c2333',
            'bg_input': '#0d1117',
            'bg_header': '#161b22',
            'fg_primary': '#ffffff',  # Branco puro para melhor contraste
            'fg_secondary': '#c9d1d9',  # Cinza claro
            'fg_table': '#f0f6fc',  # Branco azulado
            'border': '#30363d',
            'btn_primary': '#238636',
            'btn_primary_hover': '#2ea043',
            'btn_secondary': '#1f6feb',
            'btn_secondary_hover': '#388bfd',
            'btn_purple': '#8957e5',
            'btn_purple_hover': '#ab7df8',
            'btn_danger': '#da3633',
            'btn_danger_hover': '#f85149',
            'danger': '#f85149',
            'success': '#3fb950',
            'warning': '#d29922',
            'row_risco': '#2d1b1b',
            'row_risco_hover': '#3d2525',
            'row_ativo': '#0d1f0d',
            'row_ativo_hover': '#1a2e1a',
            'badge_risco': '#f85149',
            'badge_ativo': '#3fb950',
        }
        
        # Cores do tema Light
        self.cores_light = {
            'bg_primary': '#f6f8fa',
            'bg_secondary': '#ffffff',
            'bg_tertiary': '#f0f4f8',
            'bg_card': '#ffffff',
            'bg_input': '#ffffff',
            'bg_header': '#f6f8fa',
            'fg_primary': '#1f2328',
            'fg_secondary': '#656d76',
            'fg_table': '#1f2328',
            'border': '#d0d7de',
            'btn_primary': '#2da44e',
            'btn_primary_hover': '#2c974b',
            'btn_secondary': '#1f6feb',
            'btn_secondary_hover': '#388bfd',
            'btn_purple': '#8957e5',
            'btn_purple_hover': '#ab7df8',
            'btn_danger': '#cf222e',
            'btn_danger_hover': '#a0111f',
            'danger': '#cf222e',
            'success': '#2da44e',
            'warning': '#9a6700',
            'row_risco': '#fce4e4',
            'row_risco_hover': '#f5d0d0',
            'row_ativo': '#e6f5e6',
            'row_ativo_hover': '#d0e8d0',
            'badge_risco': '#cf222e',
            'badge_ativo': '#2da44e',
        }
        
        self.cores = self.cores_dark
        self.root.configure(bg=self.cores['bg_primary'])
        
        # Variáveis de controle
        self.df_resultado = None
        self.caminho_arquivo = tk.StringVar()
        self.status_texto = tk.StringVar(value="✅ Sistema pronto. Selecione uma planilha.")
        
        # Montar a interface
        self.criar_widgets()
        
    def alternar_tema(self):
        """Alterna entre tema Dark e Light"""
        if self.tema_atual == "dark":
            self.cores = self.cores_light
            self.tema_atual = "light"
            self.btn_tema.config(text="🌙 Dark", bg=self.cores['btn_secondary'])
        else:
            self.cores = self.cores_dark
            self.tema_atual = "dark"
            self.btn_tema.config(text="☀️ Light", bg=self.cores['btn_secondary'])
        
        # Atualizar cores da interface
        self.atualizar_cores()
    
    def atualizar_cores(self):
        """Atualiza todas as cores da interface"""
        self.root.configure(bg=self.cores['bg_primary'])
        
        # Atualizar cabeçalho
        self.header.configure(bg=self.cores['bg_header'])
        self.titulo.configure(bg=self.cores['bg_header'], fg=self.cores['fg_primary'])
        self.subtitulo.configure(bg=self.cores['bg_header'], fg=self.cores['fg_secondary'])
        
        # Atualizar boxes
        for widget in self.root.winfo_children():
            self._atualizar_cores_recursivo(widget)
        
        # Atualizar tabela
        try:
            style = ttk.Style()
            style.theme_use('clam')
            style.configure("Treeview", 
                          background=self.cores['bg_secondary'],
                          foreground=self.cores['fg_table'],
                          fieldbackground=self.cores['bg_secondary'])
            style.configure("Treeview.Heading",
                          background=self.cores['bg_header'],
                          foreground=self.cores['fg_primary'])
            style.map('Treeview', background=[('selected', '#1f6feb')])
        except:
            pass
    
    def _atualizar_cores_recursivo(self, widget):
        """Atualiza cores recursivamente"""
        try:
            if isinstance(widget, tk.Frame):
                widget.configure(bg=self.cores.get('bg_card', self.cores['bg_primary']))
            elif isinstance(widget, tk.LabelFrame):
                widget.configure(bg=self.cores.get('bg_card', self.cores['bg_primary']), 
                               fg=self.cores['fg_primary'])
            elif isinstance(widget, tk.Label):
                if 'bg' in widget.keys():
                    widget.configure(bg=self.cores.get('bg_card', self.cores['bg_primary']))
                if 'fg' in widget.keys():
                    widget.configure(fg=self.cores['fg_primary'])
            elif isinstance(widget, tk.Entry):
                widget.configure(bg=self.cores['bg_input'], fg=self.cores['fg_primary'])
        except:
            pass
        
        for child in widget.winfo_children():
            self._atualizar_cores_recursivo(child)
    
    def criar_widgets(self):
        """Cria todos os widgets da interface"""
        try:
            # ===== CABEÇALHO =====
            self.header = tk.Frame(self.root, bg=self.cores['bg_header'], height=90)
            self.header.pack(fill=tk.X, pady=0)
            self.header.pack_propagate(False)
            
            header_inner = tk.Frame(self.header, bg=self.cores['bg_header'])
            header_inner.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
            
            title_frame = tk.Frame(header_inner, bg=self.cores['bg_header'])
            title_frame.pack(side=tk.LEFT)
            
            self.titulo = tk.Label(title_frame, 
                                 text="🔮 Sistema de Predição de Churn",
                                 font=("Segoe UI", 20, "bold"),
                                 bg=self.cores['bg_header'],
                                 fg=self.cores['fg_primary'])
            self.titulo.pack(anchor=tk.W)
            
            self.subtitulo = tk.Label(title_frame,
                                    text="Insira uma planilha com dados dos clientes e receba a probabilidade de churn",
                                    font=("Segoe UI", 10),
                                    bg=self.cores['bg_header'],
                                    fg=self.cores['fg_secondary'])
            self.subtitulo.pack(anchor=tk.W)
            
            theme_frame = tk.Frame(header_inner, bg=self.cores['bg_header'])
            theme_frame.pack(side=tk.RIGHT)
            
            self.btn_tema = tk.Button(theme_frame,
                                     text="☀️ Light",
                                     command=self.alternar_tema,
                                     bg=self.cores['btn_secondary'],
                                     fg="white",
                                     font=("Segoe UI", 10, "bold"),
                                     relief=tk.FLAT,
                                     padx=15,
                                     pady=6,
                                     cursor="hand2")
            self.btn_tema.pack()
            
            # ===== ÁREA PRINCIPAL =====
            main_frame = tk.Frame(self.root, bg=self.cores['bg_primary'])
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # ---- Painel Esquerdo ----
            left_frame = tk.Frame(main_frame, bg=self.cores['bg_primary'], width=420)
            left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
            left_frame.pack_propagate(False)
            
            # Box Selecionar Arquivo
            box_arquivo = tk.LabelFrame(left_frame, 
                                       text=" 📂 Selecionar Planilha ",
                                       font=("Segoe UI", 11, "bold"),
                                       bg=self.cores['bg_card'],
                                       fg=self.cores['fg_primary'],
                                       bd=2,
                                       relief=tk.GROOVE)
            box_arquivo.pack(fill=tk.X, pady=(0, 15))
            
            tk.Label(box_arquivo, 
                    text="Arquivo:",
                    bg=self.cores['bg_card'],
                    fg=self.cores['fg_secondary']).pack(anchor=tk.W, padx=10, pady=(10, 0))
            
            entry_arquivo = tk.Entry(box_arquivo,
                                   textvariable=self.caminho_arquivo,
                                   font=("Segoe UI", 9),
                                   bg=self.cores['bg_input'],
                                   fg=self.cores['fg_primary'],
                                   relief=tk.FLAT,
                                   width=45)
            entry_arquivo.pack(padx=10, pady=(2, 10), fill=tk.X)
            
            btn_frame = tk.Frame(box_arquivo, bg=self.cores['bg_card'])
            btn_frame.pack(padx=10, pady=(0, 10))
            
            btn_buscar = tk.Button(btn_frame,
                                  text="📁 Buscar Planilha",
                                  command=self.selecionar_arquivo,
                                  bg=self.cores['btn_secondary'],
                                  fg="white",
                                  font=("Segoe UI", 10, "bold"),
                                  relief=tk.FLAT,
                                  padx=15,
                                  pady=8,
                                  cursor="hand2")
            btn_buscar.pack(side=tk.LEFT, padx=(0, 10))
            
            btn_processar = tk.Button(btn_frame,
                                     text="⚡ Executar Predição",
                                     command=self.executar_predicao,
                                     bg=self.cores['btn_primary'],
                                     fg="white",
                                     font=("Segoe UI", 10, "bold"),
                                     relief=tk.FLAT,
                                     padx=15,
                                     pady=8,
                                     cursor="hand2")
            btn_processar.pack(side=tk.LEFT)
            
            # Box Estatísticas
            box_stats = tk.LabelFrame(left_frame,
                                     text=" 📊 Estatísticas da Análise ",
                                     font=("Segoe UI", 11, "bold"),
                                     bg=self.cores['bg_card'],
                                     fg=self.cores['fg_primary'],
                                     bd=2,
                                     relief=tk.GROOVE)
            box_stats.pack(fill=tk.X, pady=10)
            
            self.lbl_total = tk.Label(box_stats,
                                     text="📊 Total de Clientes: --",
                                     bg=self.cores['bg_card'],
                                     fg=self.cores['fg_primary'],
                                     font=("Segoe UI", 10))
            self.lbl_total.pack(anchor=tk.W, padx=10, pady=2)
            
            self.lbl_risco = tk.Label(box_stats,
                                     text="🔴 Clientes em Risco: --",
                                     bg=self.cores['bg_card'],
                                     fg=self.cores['danger'],
                                     font=("Segoe UI", 10, "bold"))
            self.lbl_risco.pack(anchor=tk.W, padx=10, pady=2)
            
            self.lbl_ativo = tk.Label(box_stats,
                                     text="🟢 Clientes Ativos: --",
                                     bg=self.cores['bg_card'],
                                     fg=self.cores['success'],
                                     font=("Segoe UI", 10, "bold"))
            self.lbl_ativo.pack(anchor=tk.W, padx=10, pady=2)
            
            # ---- Box Ações ----
            box_acoes = tk.LabelFrame(left_frame,
                                     text=" ⚙️ Ações ",
                                     font=("Segoe UI", 11, "bold"),
                                     bg=self.cores['bg_card'],
                                     fg=self.cores['fg_primary'],
                                     bd=2,
                                     relief=tk.GROOVE)
            box_acoes.pack(fill=tk.X, pady=10)
            
            # Frame para botões com largura igual
            btn_acoes_frame = tk.Frame(box_acoes, bg=self.cores['bg_card'])
            btn_acoes_frame.pack(padx=10, pady=10, fill=tk.X)
            
            # Configurar grid para largura igual
            btn_acoes_frame.columnconfigure(0, weight=1)
            btn_acoes_frame.columnconfigure(1, weight=1)
            
            # Botão Salvar
            self.btn_export = tk.Button(btn_acoes_frame,
                                       text="📥 Salvar Resultados (Excel)",
                                       command=self.salvar_resultados,
                                       bg=self.cores['btn_purple'],
                                       fg="white",
                                       font=("Segoe UI", 10, "bold"),
                                       relief=tk.FLAT,
                                       padx=15,
                                       pady=8,
                                       cursor="hand2",
                                       state=tk.DISABLED)
            self.btn_export.grid(row=0, column=0, sticky="ew", padx=(0, 5))
            
            # Botão Limpar - mesma largura do Salvar
            self.btn_limpar = tk.Button(btn_acoes_frame,
                                       text="🗑️ Limpar Resultados",
                                       command=self.limpar_resultados,
                                       bg=self.cores['btn_danger'],
                                       fg="white",
                                       font=("Segoe UI", 10, "bold"),
                                       relief=tk.FLAT,
                                       padx=15,
                                       pady=8,
                                       cursor="hand2",
                                       state=tk.DISABLED)
            self.btn_limpar.grid(row=0, column=1, sticky="ew", padx=(5, 0))
            
            # Status
            frame_status = tk.Frame(left_frame, bg=self.cores['bg_primary'])
            frame_status.pack(fill=tk.X, pady=(20, 0))
            
            tk.Label(frame_status,
                    text="📌 Status:",
                    bg=self.cores['bg_primary'],
                    fg=self.cores['fg_secondary']).pack(anchor=tk.W)
            
            self.lbl_status = tk.Label(frame_status,
                                 textvariable=self.status_texto,
                                 bg=self.cores['bg_card'],
                                 fg=self.cores['fg_primary'],
                                 font=("Segoe UI", 9),
                                 relief=tk.FLAT,
                                 padx=10,
                                 pady=8,
                                 anchor=tk.W)
            self.lbl_status.pack(fill=tk.X, pady=(2, 0))
            
            # ---- Painel Direito (Tabela) ----
            right_frame = tk.Frame(main_frame, bg=self.cores['bg_primary'])
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            table_frame = tk.Frame(right_frame, bg=self.cores['bg_primary'])
            table_frame.pack(fill=tk.BOTH, expand=True)
            
            # Treeview
            colunas = ('Índice', 'Idade', 'Total Itens', 'Devoluções', 
                      'Pedidos', 'Gasto Total', 'Ticket Médio', 
                      'Prob. Churn', 'Status', 'Recomendação')
            
            style = ttk.Style()
            style.theme_use('clam')
            style.configure("Treeview", 
                          background=self.cores['bg_secondary'],
                          foreground=self.cores['fg_table'],
                          fieldbackground=self.cores['bg_secondary'],
                          font=("Segoe UI", 10))
            style.configure("Treeview.Heading",
                          background=self.cores['bg_header'],
                          foreground=self.cores['fg_primary'],
                          font=("Segoe UI", 10, "bold"))
            style.map('Treeview', background=[('selected', '#1f6feb')])
            
            self.tree = ttk.Treeview(table_frame, columns=colunas, show='headings', height=20)
            
            larguras = {'Índice': 55, 'Idade': 65, 'Total Itens': 95, 
                       'Devoluções': 85, 'Pedidos': 75, 'Gasto Total': 110, 
                       'Ticket Médio': 100, 'Prob. Churn': 105, 
                       'Status': 140, 'Recomendação': 160}
            
            for col in colunas:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=larguras.get(col, 80), anchor=tk.CENTER)
            
            scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
            scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
            self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
            
            self.tree.grid(row=0, column=0, sticky='nsew')
            scroll_y.grid(row=0, column=1, sticky='ns')
            scroll_x.grid(row=1, column=0, sticky='ew')
            
            table_frame.grid_rowconfigure(0, weight=1)
            table_frame.grid_columnconfigure(0, weight=1)
            
            # Rodapé
            footer = tk.Frame(self.root, bg=self.cores['bg_header'], height=35)
            footer.pack(side=tk.BOTTOM, fill=tk.X)
            
            info_modelo = tk.Label(footer,
                                 text=f"Modelo: {pipeline['modelo_nome']} | Threshold: {threshold:.2f} | Features: {len(features_esperadas)}",
                                 bg=self.cores['bg_header'],
                                 fg=self.cores['fg_secondary'],
                                 font=("Segoe UI", 9))
            info_modelo.pack(pady=8)
            
            print("✅ Interface criada com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao criar interface: {e}")
            traceback.print_exc()
    
    def selecionar_arquivo(self):
        arquivo = filedialog.askopenfilename(
            title="Selecione a planilha com os dados dos clientes",
            filetypes=[("Arquivos Excel", "*.xlsx *.xls"), ("Todos os arquivos", "*.*")]
        )
        if arquivo:
            self.caminho_arquivo.set(arquivo)
            self.status_texto.set(f"✅ Arquivo selecionado: {os.path.basename(arquivo)}")
    
    def limpar_resultados(self):
        """Limpa todos os resultados da tela"""
        # Limpar tabela
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Resetar estatísticas
        self.lbl_total.config(text="📊 Total de Clientes: --")
        self.lbl_risco.config(text="🔴 Clientes em Risco: --")
        self.lbl_ativo.config(text="🟢 Clientes Ativos: --")
        
        # Resetar status
        self.status_texto.set("🗑️ Resultados limpos. Selecione uma nova planilha.")
        
        # Desabilitar botões
        self.btn_export.config(state=tk.DISABLED)
        self.btn_limpar.config(state=tk.DISABLED)
        
        # Limpar dados
        self.df_resultado = None
    
    def executar_predicao(self):
        if not self.caminho_arquivo.get():
            messagebox.showwarning("Aviso", "Por favor, selecione um arquivo primeiro!")
            return
        
        self.status_texto.set("⏳ Processando...")
        self.root.update()
        
        try:
            df_resultado, probs = processar_planilha(self.caminho_arquivo.get())
            self.df_resultado = df_resultado
            
            total = len(df_resultado)
            risco = len(df_resultado[df_resultado['Status'] == '🔴 RISCO DE CHURN'])
            ativo = total - risco
            
            self.lbl_total.config(text=f"📊 Total de Clientes: {total}")
            self.lbl_risco.config(text=f"🔴 Clientes em Risco: {risco} ({risco/total*100:.1f}%)")
            self.lbl_ativo.config(text=f"🟢 Clientes Ativos: {ativo} ({ativo/total*100:.1f}%)")
            
            self.atualizar_tabela(df_resultado)
            self.btn_export.config(state=tk.NORMAL)
            self.btn_limpar.config(state=tk.NORMAL)
            
            self.status_texto.set(f"✅ Predição concluída! {risco} clientes em risco identificados.")
            
            if risco > total * 0.3:
                messagebox.showwarning("Aviso", 
                                      f"⚠️ Alto índice de risco identificado!\n"
                                      f"{risco} clientes ({risco/total*100:.1f}%) estão em risco de churn.")
            
        except Exception as e:
            self.status_texto.set(f"❌ Erro: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao processar:\n{str(e)}")
    
    def atualizar_tabela(self, df):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Configurar tags de cores
        self.tree.tag_configure('risco', background=self.cores['row_risco'])
        self.tree.tag_configure('ativo', background=self.cores['row_ativo'])
        
        for idx, row in df.head(1000).iterrows():
            status = row['Status']
            tag = 'risco' if status == '🔴 RISCO DE CHURN' else 'ativo'
            
            # Formatar valores
            gasto_total = row['ValorTotalGasto']
            ticket_medio = row['TicketMedio']
            
            valores = (
                idx + 1,
                row['Idade'],
                row['TotalItensComprados'],
                row['TotalItensDevolvidos'],
                row['TotalPedidos'],
                f"R$ {gasto_total:,.2f}".replace(',', '.'),
                f"R$ {ticket_medio:,.2f}".replace(',', '.'),
                f"{row['Prob_Churn_%']:.1f}%",
                status,
                row['Recomendacao']
            )
            self.tree.insert('', tk.END, values=valores, tags=(tag,))
    
    def salvar_resultados(self):
        if self.df_resultado is None:
            return
        
        nome_base = os.path.splitext(os.path.basename(self.caminho_arquivo.get()))[0]
        data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_saida = f"resultados_churn_{nome_base}_{data_hora}.xlsx"
        
        caminho_saida = filedialog.asksaveasfilename(
            title="Salvar resultados como...",
            defaultextension=".xlsx",
            filetypes=[("Arquivos Excel", "*.xlsx"), ("Todos os arquivos", "*.*")],
            initialfile=nome_saida
        )
        
        if caminho_saida:
            try:
                self.df_resultado.to_excel(caminho_saida, index=False)
                self.status_texto.set(f"✅ Resultados salvos em: {os.path.basename(caminho_saida)}")
                messagebox.showinfo("Sucesso", 
                                   f"✅ Arquivo salvo com sucesso!\n"
                                   f"📁 {os.path.basename(caminho_saida)}")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao salvar:\n{str(e)}")

# ============================================
# 5. MAIN
# ============================================
if __name__ == "__main__":
    try:
        criar_planilha_exemplo()
        
        print("🚀 Iniciando interface gráfica...")
        root = tk.Tk()
        app = ChurnApp(root)
        print("✅ Interface iniciada! Aguardando ações do usuário...")
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        traceback.print_exc()
        input("Pressione Enter para sair...")