# ==============================================================================
# PROJETO 1 - REGRESSÃO COM MLP (ESTUDO DE BASELINE E ABLAÇÃO)
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

# ------------------------------------------------------------------------------
# 1. FIXANDO SEMENTES ALEATÓRIAS
# ------------------------------------------------------------------------------
# Aqui eu fixo a semente para garantir reprodutibilidade dos resultados
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------------------
# 2. CARREGAMENTO E DIVISÃO DOS DADOS
# ------------------------------------------------------------------------------
# Leio o dataset disponibilizado
df = pd.read_csv("dataset_projeto1.csv")

X = df[['x']].values.astype(np.float32)
y = df[['y']].values.astype(np.float32)

# Divisão solicitada: 10% Treino, 10% Validação e 80% Teste
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.80, random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

print(f"[INFO] Amostras carregadas:")
print(f"       Treino:    {len(X_train)} (10%)")
print(f"       Validação: {len(X_val)} (10%)")
print(f"       Teste:     {len(X_test)} (80%)")

# DataLoaders para alimentar a rede em pequenos lotes (mini-batches)
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# ------------------------------------------------------------------------------
# 3. DEFINIÇÃO DA ARQUITETURA DA REDE MLP
# ------------------------------------------------------------------------------
# Minha rede tem 2 camadas ocultas de 64 neurônios com ativação Tanh()
class MinhaMLP(nn.Module):
    def __init__(self, taxa_dropout=0.0):
        super(MinhaMLP, self).__init__()
        self.rede = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Dropout(taxa_dropout),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Dropout(taxa_dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.rede(x)

# ------------------------------------------------------------------------------
# 4. FUNÇÃO DE TREINAMENTO E VALIDAÇÃO COM GRADIENT CLIPPING
# ------------------------------------------------------------------------------
def treinar_modelo(nome, modelo, otimizador, epocas=500, lambda_l1=0.0):
    funcao_custo = nn.MSELoss()
    perdas_treino = []
    perdas_val = []
    
    for epoca in range(epocas):
        modelo.train()
        custo_acumulado = 0.0
        
        for lote_x, lote_y in train_loader:
            otimizador.zero_grad()
            predicoes = modelo(lote_x)
            perda = funcao_custo(predicoes, lote_y)
            
            # Penalidade L1 (se configurada)
            if lambda_l1 > 0.0:
                penalidade_l1 = sum(torch.sum(torch.abs(p)) for p in modelo.parameters())
                perda += lambda_l1 * penalidade_l1
                
            perda.backward()
            
            # BLINDAGEM: Gradient Clipping para evitar explosão de gradientes (NaN) com Momentum
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            
            otimizador.step()
            custo_acumulado += perda.item() * len(lote_x)
            
        custo_medio_treino = custo_acumulado / len(X_train)
        perdas_treino.append(custo_medio_treino)
        
        # Avaliação na Validação
        modelo.eval()
        with torch.no_grad():
            pred_val = modelo(torch.tensor(X_val))
            custo_val = funcao_custo(pred_val, torch.tensor(y_val)).item()
            perdas_val.append(custo_val)
            
    return perdas_treino, perdas_val

# ------------------------------------------------------------------------------
# 5. CÁLCULO DAS MÉTRICAS (MAE, MSE, RMSE, R2)
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
# 6. CONFIGURAÇÃO DO BASELINE E ESTUDO DE ABLAÇÃO
# ------------------------------------------------------------------------------
# Usei lr=0.01 que garante estabilidade para todos os experimentos
experimentos = {
    "Baseline (SGD Puro)": {"lr": 0.01, "momentum": 0.0, "l2": 0.0,  "l1": 0.0,   "dropout": 0.0},
    "+ Momentum":          {"lr": 0.01, "momentum": 0.9, "l2": 0.0,  "l1": 0.0,   "dropout": 0.0},
    "+ L2 (Weight Decay)": {"lr": 0.01, "momentum": 0.0, "l2": 1e-3, "l1": 0.0,   "dropout": 0.0},
    "+ L1":                {"lr": 0.01, "momentum": 0.0, "l2": 0.0,  "l1": 1e-4,  "dropout": 0.0},
    "+ Dropout":           {"lr": 0.01, "momentum": 0.0, "l2": 0.0,  "l1": 0.0,   "dropout": 0.05}
}

tabela_metricas = {}
historicos = {}
modelos_treinados = {}
EPOCAS = 500

print("\n[INFO] Treinando os modelos...")

for nome_exp, config in experimentos.items():
    torch.manual_seed(SEED) # Mesma inicialização para todos
    
    modelo = MinhaMLP(taxa_dropout=config["dropout"])
    otimizador = optim.SGD(
        modelo.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["l2"]
    )
    
    treino_loss, val_loss = treinar_modelo(
        nome_exp, modelo, otimizador, epocas=EPOCAS, lambda_l1=config["l1"]
    )
    
    mae, mse, rmse, r2, _ = calcular_metricas(modelo, X_test, y_test)
    
    tabela_metricas[nome_exp] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}
    historicos[nome_exp] = (treino_loss, val_loss)
    modelos_treinados[nome_exp] = modelo
    print(f"  -> Concluído: {nome_exp}")

# ------------------------------------------------------------------------------
# 7. EXIBIÇÃO DA TABELA DE RESULTADOS
# ------------------------------------------------------------------------------
df_metricas = pd.DataFrame(tabela_metricas).T
print("\n" + "="*65)
print("TABELA COMPARATIVA FINAL (Conjunto de Teste - 80%)")
print("="*65)
print(df_metricas.round(4))
print("="*65)

# ------------------------------------------------------------------------------
# 8. GERAÇÃO DOS GRÁFICOS
# ------------------------------------------------------------------------------
print("\n[INFO] Gerando gráficos...")

# 1. Curvas de perda na validação
plt.figure(figsize=(10, 5))
for nome_exp, (_, v_loss) in historicos.items():
    plt.plot(v_loss, label=f'{nome_exp}')
plt.title("Evolução da Perda (MSE) na Validação ao Longo das Épocas")
plt.xlabel("Épocas")
plt.ylabel("MSE Loss")
plt.ylim(0, 1.5)  # Limito o eixo Y para o gráfico ficar legível
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("grafico_curvas_aprendizado.png", dpi=300)
plt.close()

# 2. Curva de predição sobre os dados reais
indices_ord = np.argsort(X.flatten())
X_plot = X[indices_ord]

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='lightgray', s=15, label='Conjunto Total (80% Teste)')
plt.scatter(X_train, y_train, color='red', s=35, label='Pontos de Treino (10%)', zorder=5)

for nome_exp in ["Baseline (SGD Puro)", "+ Momentum"]:
    mod = modelos_treinados[nome_exp]
    mod.eval()
    with torch.no_grad():
        curva_predita = mod(torch.tensor(X_plot)).numpy()
    plt.plot(X_plot, curva_predita, linewidth=2.5, label=f'Predição {nome_exp}')

plt.title("Regressão da Função: Comparação do Ajuste dos Modelos")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("grafico_ajuste_regressao.png", dpi=300)
plt.close()

print("[INFO] Gráficos salvos com sucesso na sua pasta!")