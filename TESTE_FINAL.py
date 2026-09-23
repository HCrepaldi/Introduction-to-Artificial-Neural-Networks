# ==============================================================================
# PROJETO 1 - REGRESSÃO COM MLP E ESTUDO DE ABLAÇÃO
# Arquitetura com Inicialização Espalhada (Sem Normalização)
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import copy

# ------------------------------------------------------------------------------
# 1. FIXANDO SEMENTES ALEATÓRIAS
# ------------------------------------------------------------------------------
# Fixo a semente global para que todas as execuções sejam 100% reproduzíveis.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------------------
# 2. CARREGAMENTO E DIVISÃO DOS DADOS (SEM NORMALIZAÇÃO)
# ------------------------------------------------------------------------------
# Leio o dataset original mantendo o domínio de X entre 0 e 10.
df = pd.read_csv("dataset_projeto1.csv")
X = df[['x']].values.astype(np.float32)
y = df[['y']].values.astype(np.float32)

# Divisão estrita do slide: 10% Treino, 10% Validação e 80% Teste
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.80, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

print(f"[INFO] Conjuntos particionados:")
print(f"       Treino:    {len(X_train)} amostras (10%)")
print(f"       Validação: {len(X_val)} amostras (10%)")
print(f"       Teste:     {len(X_test)} amostras (80%)\n")

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# ------------------------------------------------------------------------------
# 3. DEFINIÇÃO DA ARQUITETURA MLP COM CENTROS DISTRIBUÍDOS
# ------------------------------------------------------------------------------
# Criei esta classe com ativação GELU e inicialização distribuída no espaço [0, 10].
# Isso garante que a rede tenha neurônios dobrando ao longo de toda a curva,
# eliminando a reta horizontal nas caudas sem precisar normalizar os dados.
class MinhaMLPOtimizada(nn.Module):
    def __init__(self, taxa_dropout=0.0, n_neuronios=64):
        super(MinhaMLPOtimizada, self).__init__()
        
        self.camada1 = nn.Linear(1, n_neuronios)
        self.ativ1 = nn.GELU()
        self.drop1 = nn.Dropout(taxa_dropout)
        
        self.camada2 = nn.Linear(n_neuronios, n_neuronios)
        self.ativ2 = nn.GELU()
        self.drop2 = nn.Dropout(taxa_dropout)
        
        self.saida = nn.Linear(n_neuronios, 1)
        
        # Pulo do gato: espalho os centros dos neurônios uniformemente entre 0 e 10
        with torch.no_grad():
            centros = torch.linspace(0, 10, n_neuronios).unsqueeze(1)
            self.camada1.weight.fill_(1.5)
            self.camada1.bias.copy_(-1.5 * centros.squeeze())
            
    def forward(self, x):
        x = self.drop1(self.ativ1(self.camada1(x)))
        x = self.drop2(self.ativ2(self.camada2(x)))
        return self.saida(x)

# ------------------------------------------------------------------------------
# 4. FUNÇÃO DE TREINAMENTO COM EARLY STOPPING
# ------------------------------------------------------------------------------
# Função universal para rodar tanto o baseline quanto os aditivados.
def treinar_modelo(nome, modelo, otimizador, max_epocas=5000, paciencia=800, lambda_l1=0.0):
    criterio = nn.MSELoss()
    perdas_treino = []
    perdas_val = []
    
    melhor_loss_val = float('inf')
    melhor_pesos = None
    epoca_otima = 0
    
    for epoca in range(max_epocas):
        modelo.train()
        custo_acumulado = 0.0
        
        for lote_x, lote_y in train_loader:
            otimizador.zero_grad()
            pred = modelo(lote_x)
            perda = criterio(pred, lote_y)
            
            # Penalidade L1 explícita (se o experimento pedir)
            if lambda_l1 > 0.0:
                penalidade_l1 = sum(torch.sum(torch.abs(p)) for p in modelo.parameters())
                perda += lambda_l1 * penalidade_l1
                
            perda.backward()
            
            # Trava de segurança para não explodir gradientes
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            
            otimizador.step()
            custo_acumulado += perda.item() * len(lote_x)
            
        custo_treino = custo_acumulado / len(X_train)
        perdas_treino.append(custo_treino)
        
        # Avaliação na Validação (desliga dropout)
        modelo.eval()
        with torch.no_grad():
            pred_val = modelo(torch.tensor(X_val))
            custo_val = criterio(pred_val, torch.tensor(y_val)).item()
            perdas_val.append(custo_val)
            
        # Salva o melhor modelo antes do sobreajuste
        if custo_val < melhor_loss_val:
            melhor_loss_val = custo_val
            melhor_pesos = copy.deepcopy(modelo.state_dict())
            epoca_otima = epoca
            
        # Early Stopping: para se a validação parar de melhorar
        if epoca - epoca_otima >= paciencia:
            break
            
    # Restauro os melhores pesos encontrados na validação
    modelo.load_state_dict(melhor_pesos)
    return perdas_treino, perdas_val, epoca_otima

# ------------------------------------------------------------------------------
# 5. CÁLCULO DE MÉTRICAS NO TESTE (80%)
# ------------------------------------------------------------------------------
def calcular_metricas(modelo, X_dados, y_dados):
    modelo.eval()
    with torch.no_grad():
        y_pred = modelo(torch.tensor(X_dados)).numpy()
        
    mae = mean_absolute_error(y_dados, y_pred)
    mse = mean_squared_error(y_dados, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_dados, y_pred)
    return mae, mse, rmse, r2, y_pred

# ------------------------------------------------------------------------------
# 6. CONFIGURAÇÃO DO BASELINE E DOS ESTUDOS DE ABLAÇÃO
# ------------------------------------------------------------------------------
# Exatamente as condições exigidas pelos slides do professor:
# - Baseline: SGD puro, sem momentum, sem L1, sem L2, sem Dropout.
# - Modelos aditivados: mesma arquitetura, testando cada hiperparâmetro isolado.
experimentos = {
    "Baseline (SGD Puro)": {"lr": 0.008, "momentum": 0.0, "l2": 0.0,  "l1": 0.0,   "dropout": 0.0},
    "+ Momentum":          {"lr": 0.008, "momentum": 0.9, "l2": 0.0,  "l1": 0.0,   "dropout": 0.0},
    "+ L2 (Weight Decay)": {"lr": 0.008, "momentum": 0.0, "l2": 1e-3, "l1": 0.0,   "dropout": 0.0},
    "+ L1":                {"lr": 0.008, "momentum": 0.0, "l2": 0.0,  "l1": 1e-4,  "dropout": 0.0},
    "+ Dropout":           {"lr": 0.008, "momentum": 0.0, "l2": 0.0,  "l1": 0.0,   "dropout": 0.05}
}

tabela_metricas = {}
historicos = {}
modelos_treinados = {}

print("[INFO] Executando Baseline e Estudos de Ablação...")

for nome_exp, cfg in experimentos.items():
    # Reinicio a semente para garantir que TODOS comecem com a mesma inicialização
    torch.manual_seed(SEED)
    
    modelo = MinhaMLPOtimizada(taxa_dropout=cfg["dropout"])
    otimizador = optim.SGD(
        modelo.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["l2"]
    )
    
    perdas_tr, perdas_v, epoca_melhor = treinar_modelo(
        nome_exp, modelo, otimizador, max_epocas=4000, paciencia=700, lambda_l1=cfg["l1"]
    )
    
    mae, mse, rmse, r2, _ = calcular_metricas(modelo, X_test, y_test)
    
    tabela_metricas[nome_exp] = {
        "MAE": mae, 
        "MSE": mse, 
        "RMSE": rmse, 
        "R²": r2,
        "Melhor Época": epoca_melhor
    }
    historicos[nome_exp] = (perdas_tr, perdas_v)
    modelos_treinados[nome_exp] = modelo
    print(f"  -> {nome_exp} concluído (Parou na época {epoca_melhor}, R² = {r2:.4f})")

# ------------------------------------------------------------------------------
# 7. EXIBIÇÃO DA TABELA FINAL DE RESULTADOS
# ------------------------------------------------------------------------------
df_final = pd.DataFrame(tabela_metricas).T
print("\n" + "="*70)
print("TABELA COMPARATIVA DO ESTUDO DE ABLAÇÃO (Conjunto de Teste - 80%)")
print("="*70)
print(df_final.round(4))
print("="*70)

# ------------------------------------------------------------------------------
# 8. GERAÇÃO DOS GRÁFICOS
# ------------------------------------------------------------------------------
print("\n[INFO] Gerando gráficos comparativos...")

# Gráfico 1: Evolução da Perda de Validação
plt.figure(figsize=(11, 5))
for nome_exp, (_, v_loss) in historicos.items():
    plt.plot(v_loss[:1500], label=nome_exp) # Mostra as primeiras 1500 épocas para legibilidade
plt.title("Evolução da Perda (MSE) na Validação - Comparação da Ablação")
plt.xlabel("Épocas")
plt.ylabel("MSE Loss")
plt.ylim(0.3, 1.2)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("grafico_ablation_loss.png", dpi=300)
plt.show()

# Gráfico 2: Ajuste da Curva contra os Pontos Reais (Domínio Completo [0, 10])
indices_ord = np.argsort(X.flatten())
X_plot = X[indices_ord]

plt.figure(figsize=(14, 6))
plt.scatter(X, y, color='lightgray', s=15, label='Dados Totais (80% Teste)')
plt.scatter(X_train, y_train, color='blue', s=35, label='Pontos de Treino (10%)', zorder=5)

# Ploto o Baseline e o melhor modelo com Momentum para comparar o poder da curva
cores = {"Baseline (SGD Puro)": "teal", "+ Momentum": "crimson"}
for nome_exp in ["Baseline (SGD Puro)", "+ Momentum"]:
    mod = modelos_treinados[nome_exp]
    mod.eval()
    with torch.no_grad():
        curva = mod(torch.tensor(X_plot)).numpy()
    plt.plot(X_plot, curva, linewidth=2.5, label=f'Predição: {nome_exp}', color=cores[nome_exp])

plt.title("Regressão Não Linear com Neurônios Distribuídos de 0 a 10 (Sem Normalização)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("grafico_ablation_fit.png", dpi=300)
plt.show()

print("[OK] Gráficos 'grafico_ablation_loss.png' e 'grafico_ablation_fit.png' salvos!")