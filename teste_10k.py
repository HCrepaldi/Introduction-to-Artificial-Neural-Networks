import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. GERANDO OS 10.000 PONTOS
# ------------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

N_PONTOS = 10000
x_valores = np.linspace(0, 10, N_PONTOS)

# Mesma função original com componente harmônico
y_puro = (
    0.8 * np.sin(0.8 * x_valores) 
    + 0.6 * np.cos(3.8 * x_valores) 
    + 0.3 * np.sin(7.0 * x_valores)
)
ruido = np.random.normal(0, 0.2, N_PONTOS)
y_valores = y_puro + ruido

# Salva o arquivo CSV limpo
df_10k = pd.DataFrame({"x": x_valores, "y": y_valores})
df_10k.to_csv("dataset_10k.csv", index=False)
print(f"[OK] 'dataset_10k.csv' gerado!")

# ------------------------------------------------------------------------------
# 2. O PULO DO GATO: NORMALIZAÇÃO DE X (Evita saturação da Tanh)
# ------------------------------------------------------------------------------
# Dividimos X por 10 para ficar estritamente no intervalo [0, 1]
X_norm = (x_valores / 10.0).reshape(-1, 1).astype(np.float32)
y = y_valores.reshape(-1, 1).astype(np.float32)

X_train, X_temp, y_train, y_temp = train_test_split(X_norm, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ------------------------------------------------------------------------------
# 3. REDE MLP (64 neurônios por camada)
# ------------------------------------------------------------------------------
class MinhaMLP(nn.Module):
    def __init__(self):
        super(MinhaMLP, self).__init__()
        self.rede = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.rede(x)

modelo = MinhaMLP()
# SGD com momentum e taxa adequada para dados normalizados
otimizador = optim.SGD(modelo.parameters(), lr=0.08, momentum=0.9)
criterio = nn.MSELoss()

# ------------------------------------------------------------------------------
# 4. TREINAMENTO
# ------------------------------------------------------------------------------
print("\n[INFO] Treinando a rede com X normalizado...")
EPOCAS = 180

for epoca in range(EPOCAS):
    modelo.train()
    for lote_x, lote_y in train_loader:
        otimizador.zero_grad()
        pred = modelo(lote_x)
        loss = criterio(pred, lote_y)
        loss.backward()
        otimizador.step()
        
    if (epoca + 1) % 30 == 0:
        modelo.eval()
        with torch.no_grad():
            pred_val = modelo(torch.tensor(X_val))
            val_loss = criterio(pred_val, torch.tensor(y_val)).item()
        print(f"  Época [{epoca+1}/{EPOCAS}] - Loss de Validação: {val_loss:.4f}")

# ------------------------------------------------------------------------------
# 5. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ------------------------------------------------------------------------------
modelo.eval()
with torch.no_grad():
    y_pred_teste = modelo(torch.tensor(X_test)).numpy()
    
mse_final = mean_squared_error(y_test, y_pred_teste)
r2_final = r2_score(y_test, y_pred_teste)

print("\n" + "="*50)
print(f"RESULTADO FINAL:")
print(f"  MSE no Teste: {mse_final:.4f}")
print(f"  R² no Teste:  {r2_final:.4f}  <--- (Excelente ajuste!)")
print("="*50)

# ------------------------------------------------------------------------------
# 6. GRÁFICO VISUAL DO AJUSTE
# ------------------------------------------------------------------------------
with torch.no_grad():
    # Passamos o X normalizado para a rede prever
    curva_aprendida = modelo(torch.tensor(X_norm)).numpy()

plt.figure(figsize=(13, 6))
# Plotamos contra o x_valores original (0 a 10) para visualização correta
plt.scatter(x_valores[::10], y_valores[::10], color='lightgray', s=10, label='Dados Reais (com ruído)')
plt.plot(x_valores, y_puro, color='black', linestyle='--', linewidth=1.8, label='Função Teórica Perfeita')
plt.plot(x_valores, curva_aprendida, color='crimson', linewidth=2.5, label='Curva Aprendida pela Rede Neural')

plt.title(f"Ajuste da MLP com 10.000 pontos (R² = {r2_final:.3f}) - Sem Saturação nas Bordas", fontsize=14)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("resultado_10k_sem_saturacao.png", dpi=300)
plt.show()

print("[OK] Gráfico atualizado salvo como 'resultado_10k_sem_saturacao.png'!")