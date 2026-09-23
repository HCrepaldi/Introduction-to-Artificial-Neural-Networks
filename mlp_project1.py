# ==============================================================================
# PROJETO 1 - REGRESSAO COM MLP (BASELINE VANILLA + ESTUDO DE ABLACAO)
# ------------------------------------------------------------------------------
# Disciplina: Introducao as Redes Neurais Artificiais
# Autor: Henrique Crepaldi (RA: 120410)
#
# Objetivo deste script (versao pos-entrega, com foco DIDATICO):
#   Entender O QUE acontece com o baseline (R2 negativo -> subajuste),
#   COMO acontece (a Tanh satura no dominio [0, 10]) e
#   COMO resolver (padronizacao da entrada SEM vazamento: fit so no treino).
#
# Requisitos do professor respeitados:
#   - Tarefa de regressao com o dataset compartilhado (dataset_projeto1.csv)
#   - Split 10% treino / 10% validacao / 80% teste
#   - Baseline VANILLA: MLP basica, SGD PURO (sem Adam), sem momentum,
#     sem regularizacao, sem gradient clipping, sem init manual "esperta".
#   - Ablacao: L1, L2, dropout, momentum avaliados ISOLADAMENTE, com a
#     MESMA arquitetura e o MESMO protocolo de treino do baseline.
#   - Metricas: MAE, MSE, RMSE, R2.
#   - Graficos da evolucao treino/validacao.
# ==============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. REPRODUTIBILIDADE
# ------------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hiperparametros do baseline.
# NOTA DIDATICA (learning rate): a entrega original usava lr=0.01 e 500 epocas,
# e o baseline mal saia do lugar. Ao aumentar epocas percebeu-se que o gargalo
# era CONVERGENCIA, nao a arquitetura. Uma primeira correcao usou lr=0.05, o que
# destravou o aprendizado MAS deixou a curva de perda muito instavel (serrilhada,
# com picos): passo grande + batch pequeno (10) faz o SGD passar do ponto
# (overshoot) e "quicar" em torno do minimo, em vez de descer suave.
#
# Uma varredura de lr (ver relatorio) mediu o "jitter" (desvio das oscilacoes
# epoca-a-epoca) e mostrou o trade-off:
#     lr=0.05 -> jitter 0.030 (curva feia)   ; lr=0.01 -> jitter 0.017 (~45% menor)
# Adotou-se lr=0.01 como equilibrio: treino visivelmente mais estavel, mantendo
# convergencia saudavel. Passo e numero de epocas NAO fazem parte da arquitetura,
# entao ajusta-los nao viola a restricao "os aditivados nao mudam a arquitetura".
LR = 0.01
EPOCAS = 3000
BATCH_SIZE = 10
N_NEURONIOS = 64


# ------------------------------------------------------------------------------
# 2. CARREGAMENTO E PARTICAO DOS DADOS (10% / 10% / 80%)
# ------------------------------------------------------------------------------
def carregar_e_particionar(caminho_csv="dataset_projeto1.csv"):
    """Le o dataset e devolve os splits treino/validacao/teste em float32.

    O particionamento e feito ANTES de qualquer calculo de estatistica, para
    que a padronizacao (aplicada depois) nao sofra vazamento de informacao do
    conjunto de teste para o treino.
    """
    df = pd.read_csv(caminho_csv)
    X = df[["x"]].values.astype(np.float32)
    y = df[["y"]].values.astype(np.float32)

    # 80% para teste; os 20% restantes sao divididos meio a meio (10% / 10%).
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.80, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------------------------------------
# 3. PADRONIZACAO SEM VAZAMENTO
# ------------------------------------------------------------------------------
# POR QUE PADRONIZAR? A entrada x vive em [0, 10]. A funcao Tanh satura
# (fica "grudada" em +-1) para |z| grande, e com pesos iniciais pequenos o
# gradiente que chega na primeira camada quase zera -> a rede mal aprende.
# Centralizar x (media 0, desvio 1) mantem as ativacoes na regiao linear da
# Tanh, onde o gradiente flui. O professor permite normalizar DESDE QUE
# justificado; aqui a media/desvio sao calculados APENAS no treino (sem
# vazamento) e reaplicados em validacao e teste.
class PadronizadorTreino:
    def __init__(self, X_train):
        self.media = X_train.mean(axis=0, keepdims=True)
        self.desvio = X_train.std(axis=0, keepdims=True) + 1e-8

    def aplicar(self, X):
        return ((X - self.media) / self.desvio).astype(np.float32)


# ------------------------------------------------------------------------------
# 4. ARQUITETURA DA MLP (identica para baseline e modelos aditivados)
# ------------------------------------------------------------------------------
# Entrada(1) -> Densa(64) -> Tanh -> Densa(64) -> Tanh -> Saida(1)
# Inicializacao PADRAO do PyTorch (sem "pulo do gato"): o baseline precisa ser
# honesto. O unico componente opcional e o Dropout, que fica p=0.0 no baseline
# e nos modelos que nao o utilizam (Dropout(0.0) e identidade / no-op).
class MLP(nn.Module):
    def __init__(self, n_neuronios=N_NEURONIOS, dropout=0.0):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(1, n_neuronios),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(n_neuronios, n_neuronios),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(n_neuronios, 1),
        )

    def forward(self, x):
        return self.rede(x)


# ------------------------------------------------------------------------------
# 5. TREINAMENTO (protocolo unico e justo para todos os modelos)
# ------------------------------------------------------------------------------
def treinar(modelo, otimizador, train_loader, X_val_t, y_val_t,
            n_train, epocas=EPOCAS, lambda_l1=0.0):
    """Treina o modelo por um numero fixo de epocas (sem early stopping, para
    que a comparacao entre baseline e aditivados seja justa).

    Retorna os historicos de perda de treino e de validacao (MSE puro, sem a
    penalidade L1, para que as curvas sejam comparaveis entre si).
    """
    criterio = nn.MSELoss()
    hist_treino, hist_val = [], []

    for _ in range(epocas):
        modelo.train()
        soma_mse = 0.0
        for bx, by in train_loader:
            otimizador.zero_grad()
            pred = modelo(bx)
            perda = criterio(pred, by)

            # Perda REAL de treino (so MSE), registrada antes de somar L1.
            soma_mse += perda.item() * len(bx)

            # Penalidade L1 aplicada SOMENTE aos pesos (nunca aos bias).
            if lambda_l1 > 0.0:
                l1 = sum(
                    p.abs().sum()
                    for nome, p in modelo.named_parameters()
                    if "weight" in nome
                )
                perda = perda + lambda_l1 * l1

            perda.backward()
            otimizador.step()

        hist_treino.append(soma_mse / n_train)

        modelo.eval()
        with torch.no_grad():
            hist_val.append(criterio(modelo(X_val_t), y_val_t).item())

    return hist_treino, hist_val


def melhor_epoca(hist_val):
    """Epoca em que a perda de validacao foi minima (indicador de overfitting:
    se ela ocorre bem antes do fim, treinar mais so piora a generalizacao)."""
    return int(np.argmin(hist_val))


# ------------------------------------------------------------------------------
# 6. METRICAS (MAE, MSE, RMSE, R2)
# ------------------------------------------------------------------------------
def metricas(modelo, X_t, y_np):
    modelo.eval()
    with torch.no_grad():
        y_pred = modelo(X_t).numpy()
    mse = mean_squared_error(y_np, y_pred)
    return {
        "MAE": mean_absolute_error(y_np, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_np, y_pred),
    }


# ------------------------------------------------------------------------------
# 7. UTILITARIO: treina UM modelo do zero (semente fixa => mesma inicializacao)
# ------------------------------------------------------------------------------
def rodar_experimento(cfg, X_train_np, y_train_np, X_val_np, y_val_np):
    """Cria a MLP, o otimizador SGD e treina segundo a configuracao cfg."""
    torch.manual_seed(SEED)  # todos partem da MESMA inicializacao de pesos

    modelo = MLP(dropout=cfg["dropout"])
    otimizador = optim.SGD(
        modelo.parameters(),
        lr=LR,
        momentum=cfg["momentum"],
        weight_decay=cfg["l2"],  # weight_decay == regularizacao L2 no SGD
    )

    X_train_t = torch.tensor(X_train_np)
    y_train_t = torch.tensor(y_train_np)
    X_val_t = torch.tensor(X_val_np)
    y_val_t = torch.tensor(y_val_np)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    hist_tr, hist_val = treinar(
        modelo, otimizador, loader, X_val_t, y_val_t,
        n_train=len(X_train_np), lambda_l1=cfg["l1"],
    )
    return modelo, hist_tr, hist_val


# ==============================================================================
# 8. EXECUCAO PRINCIPAL
# ==============================================================================
def main():
    X_train, X_val, X_test, y_train, y_val, y_test = carregar_e_particionar()

    print("[INFO] Particao dos dados:")
    print(f"       Treino:    {len(X_train)} amostras (10%)")
    print(f"       Validacao: {len(X_val)} amostras (10%)")
    print(f"       Teste:     {len(X_test)} amostras (80%)\n")

    # --------------------------------------------------------------------------
    # 8.1. DEMONSTRACAO DIDATICA: o baseline SEM padronizacao (reproduz a
    #      entrega original) vs. o baseline COM padronizacao.
    # --------------------------------------------------------------------------
    cfg_baseline = {"momentum": 0.0, "l2": 0.0, "l1": 0.0, "dropout": 0.0}

    print("[INFO] (1/2) Baseline SEM padronizacao (escala original [0, 10])...")
    mod_raw, hist_tr_raw, hist_val_raw = rodar_experimento(
        cfg_baseline, X_train, y_train, X_val, y_val
    )
    m_raw = metricas(mod_raw, torch.tensor(X_test), y_test)
    print(f"       -> R2 teste = {m_raw['R2']:.4f} (subajuste: Tanh satura)\n")

    print("[INFO] (2/2) Baseline COM padronizacao (fit so no treino)...")
    padr = PadronizadorTreino(X_train)
    Xtr_s = padr.aplicar(X_train)
    Xval_s = padr.aplicar(X_val)
    Xtest_s = padr.aplicar(X_test)

    mod_std, hist_tr_std, hist_val_std = rodar_experimento(
        cfg_baseline, Xtr_s, y_train, Xval_s, y_val
    )
    m_std = metricas(mod_std, torch.tensor(Xtest_s), y_test)
    print(f"       -> R2 teste = {m_std['R2']:.4f} (agora a rede aprende)\n")

    # --------------------------------------------------------------------------
    # 8.2. ESTUDO DE ABLACAO (sobre os dados PADRONIZADOS)
    #      Mesma arquitetura e protocolo; muda-se UM componente por vez.
    # --------------------------------------------------------------------------
    experimentos = {
        "Baseline (SGD puro)":  {"momentum": 0.0, "l2": 0.0,  "l1": 0.0,   "dropout": 0.0},
        "+ Momentum (0.9)":     {"momentum": 0.9, "l2": 0.0,  "l1": 0.0,   "dropout": 0.0},
        "+ L2 (1e-3)":          {"momentum": 0.0, "l2": 1e-3, "l1": 0.0,   "dropout": 0.0},
        "+ L1 (1e-4)":          {"momentum": 0.0, "l2": 0.0,  "l1": 1e-4,  "dropout": 0.0},
        "+ Dropout (0.05)":     {"momentum": 0.0, "l2": 0.0,  "l1": 0.0,   "dropout": 0.05},
    }

    print("[INFO] Estudo de ablacao (dados padronizados)...")
    tabela, historicos, modelos = {}, {}, {}
    for nome, cfg in experimentos.items():
        modelo, hist_tr, hist_val = rodar_experimento(
            cfg, Xtr_s, y_train, Xval_s, y_val
        )
        met = metricas(modelo, torch.tensor(Xtest_s), y_test)
        met["Melhor epoca (val)"] = melhor_epoca(hist_val)
        tabela[nome] = met
        historicos[nome] = (hist_tr, hist_val)
        modelos[nome] = modelo
        print(f"       -> {nome:22s} R2 = {met['R2']:.4f}  "
              f"(melhor val na epoca {met['Melhor epoca (val)']})")

    # --------------------------------------------------------------------------
    # 8.3. TABELAS DE RESULTADOS
    # --------------------------------------------------------------------------
    df_antes_depois = pd.DataFrame(
        {"Sem padronizacao": m_raw, "Com padronizacao": m_std}
    ).T
    print("\n" + "=" * 70)
    print("EFEITO DA PADRONIZACAO NO BASELINE (Conjunto de Teste - 80%)")
    print("=" * 70)
    print(df_antes_depois.round(4))

    df_abl = pd.DataFrame(tabela).T
    print("\n" + "=" * 70)
    print("ESTUDO DE ABLACAO (Conjunto de Teste - 80%, dados padronizados)")
    print("=" * 70)
    print(df_abl.round(4))
    print("=" * 70 + "\n")

    # --------------------------------------------------------------------------
    # 8.4. GRAFICOS
    # --------------------------------------------------------------------------
    gerar_graficos(
        X_train, y_train, X_test, y_test, padr,
        hist_tr_raw, hist_val_raw, hist_tr_std, hist_val_std,
        historicos, modelos,
    )
    print("[OK] Graficos salvos com sucesso.")


# ------------------------------------------------------------------------------
# 9. GRAFICOS (didaticos)
# ------------------------------------------------------------------------------
def gerar_graficos(X_train, y_train, X_test, y_test, padr,
                   hist_tr_raw, hist_val_raw, hist_tr_std, hist_val_std,
                   historicos, modelos):
    # --- Grafico A: treino vs validacao, SEM vs COM padronizacao ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    axes[0].plot(hist_tr_raw, label="Treino")
    axes[0].plot(hist_val_raw, label="Validacao")
    axes[0].set_title("Baseline SEM padronizacao (subajuste)")
    axes[0].set_xlabel("Epocas"); axes[0].set_ylabel("MSE")
    axes[0].legend(); axes[0].grid(True, ls="--", alpha=0.5)

    axes[1].plot(hist_tr_std, label="Treino")
    axes[1].plot(hist_val_std, label="Validacao")
    axes[1].set_title("Baseline COM padronizacao (aprende)")
    axes[1].set_xlabel("Epocas")
    axes[1].legend(); axes[1].grid(True, ls="--", alpha=0.5)
    fig.suptitle("Evolucao do Treinamento (Treino vs. Validacao)")
    fig.tight_layout()
    fig.savefig("grafico_curvas_aprendizado.png", dpi=150)
    plt.close(fig)

    # --- Grafico B: comparacao da perda de validacao na ablacao ---
    plt.figure(figsize=(11, 5))
    for nome, (_, v) in historicos.items():
        plt.plot(v, label=nome)
    plt.title("Ablacao: Perda de Validacao (MSE) ao Longo das Epocas")
    plt.xlabel("Epocas"); plt.ylabel("MSE de Validacao")
    plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafico_ablation_loss.png", dpi=150)
    plt.close()

    # --- Grafico C: ajuste da curva sobre todos os dados ---
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    ordem = np.argsort(X_all.flatten())
    X_plot = X_all[ordem]
    X_plot_s = torch.tensor(padr.aplicar(X_plot))

    plt.figure(figsize=(13, 6))
    plt.scatter(X_test, y_test, color="lightgray", s=15, label="Teste (80%)")
    plt.scatter(X_train, y_train, color="red", s=40, zorder=5,
                label="Treino (10%)")
    for nome in ["Baseline (SGD puro)", "+ Momentum (0.9)"]:
        m = modelos[nome]
        m.eval()
        with torch.no_grad():
            curva = m(X_plot_s).numpy()
        plt.plot(X_plot, curva, linewidth=2.5, label=f"Predicao {nome}")
    plt.title("Regressao com MLP (entrada padronizada): ajuste dos modelos")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("grafico_ajuste_regressao.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
