import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import copy

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv("dataset_projeto1.csv")
X = df[['x']].values.astype(np.float32)
y = df[['y']].values.astype(np.float32)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.80, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# ------------------------------------------------------------------------------
# REDE COM INICIALIZAÇÃO ESPALHADA NO DOMÍNIO [0, 10]
# ------------------------------------------------------------------------------
class RedeEspalhada(nn.Module):
    def __init__(self, n_neuronios=64):
        super(RedeEspalhada, self).__init__()
        self.camada1 = nn.Linear(1, n_neuronios)
        self.ativ1 = nn.GELU()
        self.camada2 = nn.Linear(n_neuronios, n_neuronios)
        self.ativ2 = nn.GELU()
        self.saida = nn.Linear(n_neuronios, 1)
        
        # O SEGREDO: Distribuir os centros dos neurônios ao longo de [0, 10]
        with torch.no_grad():
            # Espalha os centros c uniformemente de 0 a 10
            centros = torch.linspace(0, 10, n_neuronios).unsqueeze(1)
            self.camada1.weight.fill_(1.5)               # Inclinação do cotovelo
            self.camada1.bias.copy_(-1.5 * centros.squeeze()) # Força a dobra em x = centro!
        
    def forward(self, x):
        x = self.ativ1(self.camada1(x))
        x = self.ativ2(self.camada2(x))
        return self.saida(x)

modelo = RedeEspalhada(n_neuronios=64)

# SGD com momentum e learning rate ligeiramente menor para refinar
otimizador = optim.SGD(modelo.parameters(), lr=0.008, momentum=0.9)
criterio = nn.MSELoss()

MAX_EPOCAS = 12000
PACIENCIA = 1200
melhor_loss_val = float('inf')
melhor_pesos = None
epoca_melhor = 0

historico_treino = []
historico_val = []

print("[INFO] Treinando rede com neurônios distribuídos de 0 a 10...")

for epoca in range(MAX_EPOCAS):
    modelo.train()
    loss_acum = 0.0
    for bx, by in train_loader:
        otimizador.zero_grad()
        pred = modelo(bx)
        loss = criterio(pred, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        otimizador.step()
        loss_acum += loss.item() * len(bx)
        
    l_tr = loss_acum / len(X_train)
    historico_treino.append(l_tr)
    
    modelo.eval()
    with torch.no_grad():
        val_pred = modelo(torch.tensor(X_val))
        l_val = criterio(val_pred, torch.tensor(y_val)).item()
        historico_val.append(l_val)
        
    if l_val < melhor_loss_val:
        melhor_loss_val = l_val
        melhor_pesos = copy.deepcopy(modelo.state_dict())
        epoca_melhor = epoca
        
    if epoca - epoca_melhor >= PACIENCIA:
        print(f"[PAROU] Early stopping na época {epoca}! Melhor época: {epoca_melhor}")
        break

modelo.load_state_dict(melhor_pesos)

# AVALIAÇÃO
modelo.eval()
with torch.no_grad():
    y_pred_teste = modelo(torch.tensor(X_test)).numpy()
    
r2 = r2_score(y_test, y_pred_teste)
mse = mean_squared_error(y_test, y_pred_teste)

print("\n" + "="*50)
print(f"RESULTADO FINAL:")
print(f"  MSE Teste: {mse:.4f}")
print(f"  R² Teste:  {r2:.4f}")
print("="*50)

# PLOT
indices_ord = np.argsort(X.flatten())
X_plot = X[indices_ord]

with torch.no_grad():
    curva = modelo(torch.tensor(X_plot)).numpy()

plt.figure(figsize=(13, 6))
plt.scatter(X, y, color='lightgray', s=15, label='Dados Totais (80% Teste)')
plt.scatter(X_train, y_train, color='blue', s=40, label='Treino (10%)', zorder=5)
plt.plot(X_plot, curva, color='crimson', linewidth=2.5, label='MLP com Neurônios Espalhados')
plt.title(f"Ajuste da MLP Sem Normalização - Neurônios de 0 a 10 (R² = {r2:.3f})")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("curva_desbloqueada_final.png", dpi=300)
plt.show()