# ==============================================================================
# PROJETO 3 - MODELO GENERATIVO (VAE) PARA O FASHION-MNIST
# ------------------------------------------------------------------------------
# Disciplina: Introducao as Redes Neurais Artificiais
# Autor: Henrique Crepaldi (RA: 120410)
#
# Objetivo deste script (com foco DIDATICO):
#   Treinar um modelo generativo - um AUTOENCODER VARIACIONAL (VAE) - para o
#   dataset Fashion-MNIST e EXPLORAR o espaco latente aprendido.
#
#   Conforme o enunciado, treinamos o VAE em DUAS configuracoes de espaco
#   latente e ajustamos, para cada uma, a melhor topologia por otimizacao de
#   hiperparametros guiada pela FUNCAO DE CUSTO no conjunto de VALIDACAO:
#
#     - Espaco latente 2 dimensoes  (2D)
#     - Espaco latente 8 dimensoes  (8D)
#
#   Depois, exploramos o espaco latente:
#     - Grafico da representacao latente 2D (direto, pois ja e 2D).
#     - Grafico da PROJECAO do espaco latente 8D em 2D via PCA (pois 8D nao
#       cabe num grafico plano).
#     - Em ambos, COLORIMOS os pontos pelos ROTULOS (classes de roupa).
#
#   E respondemos as perguntas do enunciado:
#     (a) Ha formacao de clusters no espaco latente?
#     (b) Ha separacao dos rotulos no espaco latente?
#     (c) A projecao (PCA) ilustra quanto da variancia?
#
# POR QUE UM VAE (E NAO UM AE PADRAO)?
#   Um AE padrao aprende uma representacao latente, mas o espaco latente dele
#   NAO e necessariamente "navegavel": ao amostrar um ponto qualquer e decodificar
#   podemos gerar lixo (slides: "Regioes nao aprendidas podem gerar exemplos
#   inadequados"). O VAE REGULARIZA o treino (termo KL) para que o espaco latente
#   se aproxime de uma Normal(0, I) - continuo e navegavel - o que o torna um
#   MODELO GENERATIVO de verdade: podemos amostrar z ~ N(0, I) e gerar roupas
#   novas plausiveis.
#
# FUNCAO DE CUSTO DO VAE (slides, "FUNCAO DE CUSTO"):
#     loss = termo_de_reconstrucao(x, x_reconstruido)
#          + termo_de_regularizacao_KL( N(mu, sigma) || N(0, I) )
#   Usamos este ELBO como criterio de selecao de modelo na VALIDACAO.
#
# TRUQUE DA REPARAMETRIZACAO (pergunta explicita do professor):
#   A amostragem z ~ N(mu, sigma) e estocastica e NAO-diferenciavel; o gradiente
#   do erro em relacao aos pesos do encoder "trava" no no de amostragem. O truque
#   reescreve z = mu + sigma * eps, com eps ~ N(0, I) FIXO (sem gradiente). Assim
#   a aleatoriedade sai para um ramo lateral (eps) e z vira uma funcao
#   DETERMINISTICA e diferenciavel de mu e sigma -> o backprop volta a fluir para
#   o encoder. Isso esta implementado em VAE.reparametrizar (comentado la).
#
# CUIDADO COM VAZAMENTO DE DADOS (data leakage) - declarado explicitamente:
#   - O conjunto de TESTE oficial do Fashion-MNIST (10k imagens) e reservado e
#     NUNCA e usado para escolher hiperparametros nem para early stopping.
#   - O ajuste de hiperparametros usa SOMENTE treino+validacao, e a validacao e
#     separada do treino ANTES de qualquer treinamento.
#   - A normalizacao e um simples /255.0 (constante global, sem estatisticas
#     aprendidas dos dados), entao nao ha vazamento por normalizacao.
#
# COMO RODAR:
#   python -m venv .venv && source .venv/bin/activate
#   # CPU:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   # GPU (ex.: RTX 3050, CUDA 12.1) - para rodar na SUA maquina:
#   # pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install numpy matplotlib scikit-learn
#   python vae_fashion_mnist_project3.py                 # roda tudo (rapido)
#   python vae_fashion_mnist_project3.py --epocas 40     # treino mais longo
#
# OBS sobre GPU: o script detecta o dispositivo automaticamente (ver DEVICE).
#   Aqui (sandbox) nao ha GPU, entao roda em CPU. Na sua maquina com a RTX 3050
#   e PyTorch-CUDA instalado, o MESMO script usa a GPU sem alterar nada.
# ==============================================================================

import argparse
import gzip
import os
import struct
import time
import urllib.request

# Matplotlib precisa de um diretorio de cache gravavel; em alguns ambientes o
# HOME e read-only. Definimos o MPLCONFIGDIR antes de importar o pyplot para
# evitar avisos e falhas. (Nao muda nenhum resultado numerico.)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem janela: salva PNG em arquivo (nao abre tela)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA


# ------------------------------------------------------------------------------
# 1. REPRODUTIBILIDADE E DISPOSITIVO (CPU/GPU)
# ------------------------------------------------------------------------------
# Mesma semente dos projetos anteriores: garante que split, inicializacoes de
# pesos, amostragens (eps do VAE) e embaralhamento dos lotes sejam repetiveis
# entre execucoes - essencial para a banca reproduzir os numeros do relatorio.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Deteccao automatica do dispositivo. Se houver GPU NVIDIA com PyTorch-CUDA,
# usa "cuda"; caso contrario, "cpu". O codigo abaixo usa sempre .to(DEVICE),
# entao o MESMO script roda nos dois cenarios sem alteracao.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nomes das 10 classes do Fashion-MNIST (rotulo 0..9), para legendas dos graficos.
NOMES_CLASSES = [
    "Camiseta/top", "Calca", "Pulover", "Vestido", "Casaco",
    "Sandalia", "Camisa", "Tenis", "Bolsa", "Bota",
]


# ==============================================================================
# 2. DADOS: BAIXAR E CARREGAR O FASHION-MNIST (SEM torchvision)
# ==============================================================================
# O Fashion-MNIST e distribuido no mesmo formato "idx" do MNIST classico: 4
# arquivos .gz (imagens e rotulos de treino e de teste). Em vez de depender do
# torchvision, baixamos os 4 arquivos direto de mirrors publicos e lemos o
# formato binario na mao - deixa o script AUTOCONTIDO e transparente.
#
# Formato idx (big-endian):
#   - imagens: [magic(4)][n(4)][linhas(4)][colunas(4)] + n*28*28 bytes (0..255)
#   - rotulos: [magic(4)][n(4)] + n bytes (0..9)

ARQUIVOS = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Mirrors tentados em ordem. O primeiro que responder e usado por arquivo.
MIRRORS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",  # fallback de formato idx
]


def _baixar_arquivo(nome_arquivo, pasta_destino):
    """Baixa um .gz do Fashion-MNIST tentando os mirrors em ordem.

    Se o arquivo ja existe localmente (cache), nao baixa de novo - isso deixa
    execucoes repetidas rapidas e nao depende da rede toda vez.
    """
    caminho_local = os.path.join(pasta_destino, nome_arquivo)
    if os.path.exists(caminho_local) and os.path.getsize(caminho_local) > 0:
        return caminho_local

    ultimo_erro = None
    for base in MIRRORS:
        url = base + nome_arquivo
        try:
            print(f"    baixando {nome_arquivo} de {base} ...")
            with urllib.request.urlopen(url, timeout=60) as resposta:
                dados = resposta.read()
            # So aceita se o conteudo parece um gzip valido (magic 0x1f8b).
            if len(dados) < 2 or dados[0] != 0x1F or dados[1] != 0x8B:
                raise ValueError("conteudo nao parece gzip")
            with open(caminho_local, "wb") as f:
                f.write(dados)
            return caminho_local
        except Exception as erro:  # tenta o proximo mirror
            ultimo_erro = erro
            print(f"      falhou neste mirror: {erro}")
    raise RuntimeError(f"nao foi possivel baixar {nome_arquivo}: {ultimo_erro}")


def _ler_idx_imagens(caminho_gz):
    """Le um arquivo idx de IMAGENS (.gz) e devolve array (n, 28, 28) uint8."""
    with gzip.open(caminho_gz, "rb") as f:
        magic, n, linhas, colunas = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"magic de imagens inesperado: {magic}")
        buffer = f.read(n * linhas * colunas)
    return np.frombuffer(buffer, dtype=np.uint8).reshape(n, linhas, colunas)


def _ler_idx_rotulos(caminho_gz):
    """Le um arquivo idx de ROTULOS (.gz) e devolve array (n,) uint8."""
    with gzip.open(caminho_gz, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"magic de rotulos inesperado: {magic}")
        buffer = f.read(n)
    return np.frombuffer(buffer, dtype=np.uint8)


def carregar_fashion_mnist(pasta_dados):
    """Garante o download e devolve os 4 arrays numpy do Fashion-MNIST.

    Retorna: (x_treino, y_treino, x_teste, y_teste)
      - x_*: (n, 28, 28) uint8 em [0, 255]
      - y_*: (n,) uint8 em [0, 9]
    """
    os.makedirs(pasta_dados, exist_ok=True)
    caminhos = {chave: _baixar_arquivo(nome, pasta_dados) for chave, nome in ARQUIVOS.items()}
    x_treino = _ler_idx_imagens(caminhos["train_images"])
    y_treino = _ler_idx_rotulos(caminhos["train_labels"])
    x_teste = _ler_idx_imagens(caminhos["test_images"])
    y_teste = _ler_idx_rotulos(caminhos["test_labels"])
    return x_treino, y_treino, x_teste, y_teste


def preparar_tensores(x_uint8, y_uint8):
    """Converte imagens uint8 (n,28,28) em tensores (n,784) float em [0,1].

    POR QUE /255.0 (E SO ISSO)?
      Os pixels vem em [0, 255]. Dividir por 255 leva tudo para [0, 1]. Essa e
      uma constante GLOBAL (nao depende de media/desvio dos dados), entao nao ha
      risco de vazamento de estatisticas do teste para o treino. Alem disso, com
      a entrada em [0, 1] podemos usar a saida do decoder com sigmoide e o erro
      de reconstrucao por entropia cruzada binaria (BCE) - padrao para VAE de
      imagens em [0, 1].
    """
    x = torch.from_numpy(x_uint8.astype(np.float32) / 255.0).reshape(-1, 28 * 28)
    y = torch.from_numpy(y_uint8.astype(np.int64))
    return x, y


# ==============================================================================
# 3. O MODELO: AUTOENCODER VARIACIONAL (VAE) EM MLP
# ==============================================================================
# Arquitetura (MLP, conforme os slides de AE padrao/profundo):
#
#   ENCODER:   x (784) -> [camadas ocultas] -> DOIS vetores de saida:
#                mu    (dim_latente)   -> media    da distribuicao latente
#                logvar(dim_latente)   -> log da variancia (ver nota abaixo)
#
#   AMOSTRAGEM (truque da reparametrizacao):
#                z = mu + sigma * eps,  eps ~ N(0, I),  sigma = exp(0.5*logvar)
#
#   DECODER:   z (dim_latente) -> [camadas ocultas espelhadas] -> x_rec (784)
#                com sigmoide na saida (pixels reconstruidos em [0, 1]).
#
# POR QUE A REDE PREVE logvar E NAO sigma DIRETO?
#   A variancia tem de ser POSITIVA. Se a rede previsse sigma direto, teriamos
#   de forcar sigma > 0 (gambiarra). Prevendo logvar (o LOG da variancia), o
#   valor pode ser qualquer real em (-inf, +inf) e recuperamos sigma de forma
#   sempre positiva e numericamente estavel com sigma = exp(0.5 * logvar).
#   (0.5 porque variancia = sigma^2  =>  sigma = exp(0.5*logvar).)

class VAE(nn.Module):
    """VAE em MLP com profundidade/largura configuraveis (para a busca de
    hiperparametros). camadas_ocultas e uma lista, ex.: [512, 256]."""

    def __init__(self, dim_entrada=784, camadas_ocultas=(512, 256), dim_latente=2):
        super().__init__()
        self.dim_entrada = dim_entrada
        self.dim_latente = dim_latente

        # ---- ENCODER: pilha de camadas Linear+ReLU ate a ultima oculta ------
        camadas_enc = []
        tamanho_anterior = dim_entrada
        for tamanho in camadas_ocultas:
            camadas_enc.append(nn.Linear(tamanho_anterior, tamanho))
            camadas_enc.append(nn.ReLU())
            tamanho_anterior = tamanho
        self.encoder = nn.Sequential(*camadas_enc)

        # Duas "cabecas" lineares que leem a ultima camada oculta e produzem,
        # respectivamente, mu e logvar (ambos com dim_latente componentes).
        self.cabeca_mu = nn.Linear(tamanho_anterior, dim_latente)
        self.cabeca_logvar = nn.Linear(tamanho_anterior, dim_latente)

        # ---- DECODER: espelha as camadas ocultas na ordem inversa -----------
        camadas_dec = []
        tamanho_anterior = dim_latente
        for tamanho in reversed(camadas_ocultas):
            camadas_dec.append(nn.Linear(tamanho_anterior, tamanho))
            camadas_dec.append(nn.ReLU())
            tamanho_anterior = tamanho
        camadas_dec.append(nn.Linear(tamanho_anterior, dim_entrada))
        camadas_dec.append(nn.Sigmoid())  # saida em [0, 1], casa com a entrada
        self.decoder = nn.Sequential(*camadas_dec)

    # --------------------------------------------------------------------------
    def codificar(self, x):
        """x (lote, 784) -> (mu, logvar), cada um (lote, dim_latente)."""
        h = self.encoder(x)
        return self.cabeca_mu(h), self.cabeca_logvar(h)

    # --------------------------------------------------------------------------
    def reparametrizar(self, mu, logvar):
        """TRUQUE DA REPARAMETRIZACAO (coracao do VAE).

        PROBLEMA: queremos z ~ N(mu, sigma^2). Amostrar diretamente desse
        Normal e uma operacao ESTOCASTICA e NAO-DIFERENCIAVEL em relacao a mu e
        sigma; o backpropagation nao consegue atravessar esse "no de sorteio" e
        o gradiente do erro NAO chega aos pesos do encoder (o fluxo trava).

        SOLUCAO: mover a aleatoriedade para fora do caminho dos parametros.
        Em vez de sortear z do Normal(mu, sigma^2), sorteamos um ruido PADRAO
        eps ~ N(0, I) (que NAO depende de nenhum peso) e montamos:

            z = mu + sigma * eps

        Agora z e uma funcao DETERMINISTICA e DIFERENCIAVEL de mu e sigma: a
        unica fonte de aleatoriedade (eps) entra como uma constante lateral,
        sem gradiente. Assim dz/dmu = 1 e dz/dsigma = eps estao bem definidos e
        o gradiente volta a fluir normalmente para o encoder.
        """
        sigma = torch.exp(0.5 * logvar)          # desvio-padrao (sempre > 0)
        eps = torch.randn_like(sigma)            # eps ~ N(0, I); SEM gradiente
        return mu + sigma * eps                  # z reparametrizado

    # --------------------------------------------------------------------------
    def decodificar(self, z):
        """z (lote, dim_latente) -> reconstrucao (lote, 784) em [0, 1]."""
        return self.decoder(z)

    # --------------------------------------------------------------------------
    def forward(self, x):
        """Passagem completa: x -> (mu, logvar) -> z -> x_reconstruido."""
        mu, logvar = self.codificar(x)
        z = self.reparametrizar(mu, logvar)
        x_rec = self.decodificar(z)
        return x_rec, mu, logvar


# ==============================================================================
# 4. FUNCAO DE CUSTO DO VAE (ELBO = reconstrucao + beta * KL)
# ==============================================================================
def custo_vae(x, x_rec, mu, logvar, beta=1.0):
    """Funcao de custo do VAE (slides: "FUNCAO DE CUSTO").

        loss = RECONSTRUCAO + beta * KL

    1) TERMO DE RECONSTRUCAO: quao fiel e a reconstrucao x_rec em relacao a x.
       Como x esta em [0, 1] e a saida do decoder usa sigmoide, usamos a
       ENTROPIA CRUZADA BINARIA (BCE) somada sobre os 784 pixels. (E o analogo
       probabilistico do "erro de reconstrucao" ||x - d(e(x))||^2 dos slides,
       mais adequado para pixels em [0, 1].)

    2) TERMO DE REGULARIZACAO (KL): empurra a distribuicao latente aproximada
       N(mu, sigma^2) para perto da Normal padrao N(0, I). Para dois gaussianos
       diagonais existe forma fechada:
           KL = -0.5 * sum( 1 + logvar - mu^2 - exp(logvar) )
       E essa regularizacao que deixa o espaco latente continuo/navegavel (o
       que torna o VAE um modelo generativo util, e nao so um compressor).

    beta: peso do termo KL (beta=1 e o VAE classico). Deixamos como
    hiperparametro porque beta controla o equilibrio reconstrucao vs. latente
    organizado (ideia do "beta-VAE"); e um dos itens da busca.

    Retorna a loss MEDIA por exemplo (somamos sobre pixels/dimensoes e tiramos a
    media sobre o lote) para que o valor nao dependa do tamanho do lote e seja
    comparavel entre treino, validacao e teste.
    """
    tamanho_lote = x.size(0)
    # BCE somada sobre os 784 pixels, depois media sobre o lote.
    recon = F.binary_cross_entropy(x_rec, x, reduction="sum") / tamanho_lote
    # KL em forma fechada, somada sobre as dimensoes latentes, media sobre o lote.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / tamanho_lote
    return recon + beta * kl, recon, kl


# ==============================================================================
# 5. TREINO E AVALIACAO
# ==============================================================================
def avaliar(modelo, carregador, beta):
    """Calcula a loss MEDIA (ELBO) do modelo sobre um carregador de dados.

    Usado para medir a loss de VALIDACAO (selecao de modelo) e de TESTE (relato
    final). Roda em modo eval() e sem gradiente (mais rapido e sem efeitos
    colaterais no treino)."""
    modelo.eval()
    soma_loss = soma_recon = soma_kl = 0.0
    n_total = 0
    with torch.no_grad():
        for (x_lote,) in carregador:
            x_lote = x_lote.to(DEVICE)
            x_rec, mu, logvar = modelo(x_lote)
            loss, recon, kl = custo_vae(x_lote, x_rec, mu, logvar, beta)
            # Re-multiplicamos pela quantidade do lote para somar corretamente e
            # dividir pelo total no fim (medias de lotes de tamanhos diferentes).
            n = x_lote.size(0)
            soma_loss += loss.item() * n
            soma_recon += recon.item() * n
            soma_kl += kl.item() * n
            n_total += n
    return soma_loss / n_total, soma_recon / n_total, soma_kl / n_total


def treinar_vae(modelo, carregador_treino, carregador_val, epocas, taxa_aprendizado,
                beta, verboso=True):
    """Treina um VAE com Adam, acompanhando a loss de validacao a cada epoca.

    Faz EARLY STOPPING pela loss de VALIDACAO: guarda os pesos da melhor epoca
    (menor loss de val) e os restaura no fim. O conjunto de teste NAO participa
    disso - evita vazamento.

    Retorna um dicionario com historico e a melhor loss de validacao."""
    modelo = modelo.to(DEVICE)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa_aprendizado)

    historico = {"treino": [], "val": [], "val_recon": [], "val_kl": []}
    melhor_val = float("inf")
    melhor_estado = None
    paciencia = 6          # epocas sem melhora antes de parar
    epocas_sem_melhora = 0

    for epoca in range(1, epocas + 1):
        # ---- uma passada de treino -----------------------------------------
        modelo.train()
        soma_loss = 0.0
        n_total = 0
        for (x_lote,) in carregador_treino:
            x_lote = x_lote.to(DEVICE)
            otimizador.zero_grad()
            x_rec, mu, logvar = modelo(x_lote)
            loss, _, _ = custo_vae(x_lote, x_rec, mu, logvar, beta)
            loss.backward()          # backprop (funciona gracas a reparametrizacao)
            otimizador.step()
            n = x_lote.size(0)
            soma_loss += loss.item() * n
            n_total += n
        loss_treino = soma_loss / n_total

        # ---- avaliacao na validacao ----------------------------------------
        loss_val, recon_val, kl_val = avaliar(modelo, carregador_val, beta)
        historico["treino"].append(loss_treino)
        historico["val"].append(loss_val)
        historico["val_recon"].append(recon_val)
        historico["val_kl"].append(kl_val)

        if verboso:
            print(f"      epoca {epoca:02d}/{epocas} | treino={loss_treino:8.3f} "
                  f"| val={loss_val:8.3f} (recon={recon_val:7.3f} kl={kl_val:6.3f})")

        # ---- early stopping pela loss de validacao -------------------------
        if loss_val < melhor_val - 1e-3:
            melhor_val = loss_val
            # guarda uma copia dos pesos da melhor epoca (em CPU, por seguranca)
            melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                if verboso:
                    print(f"      early stopping na epoca {epoca} "
                          f"(melhor val={melhor_val:.3f})")
                break

    # restaura os melhores pesos antes de devolver o modelo
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)

    return {"historico": historico, "melhor_val": melhor_val, "modelo": modelo}


# ==============================================================================
# 6. OTIMIZACAO DE HIPERPARAMETROS (SELECAO PELA LOSS DE VALIDACAO)
# ==============================================================================
# Para CADA dimensao latente (2D e 8D) procuramos a melhor TOPOLOGIA/configuracao
# por uma busca em grade pequena. O criterio de selecao e a LOSS no conjunto de
# VALIDACAO (o enunciado pede exatamente isso). O conjunto de teste fica fora.
#
# A grade combina:
#   - topologia (camadas ocultas): rede mais rasa vs. mais profunda
#   - beta (peso do termo KL): equilibrio reconstrucao vs. latente regularizado
#   - taxa de aprendizado
#
# Mantemos a grade enxuta de proposito: o objetivo e DIDATICO e tem de rodar em
# tempo razoavel (inclusive em CPU). Na sua maquina com GPU da para ampliar.

def grade_hiperparametros():
    """Devolve a lista de configuracoes testadas na busca (mesma grade para 2D
    e 8D; a arquitetura PODE variar por dimensao porque a busca escolhe
    independentemente a melhor para cada uma)."""
    topologias = [
        ("rasa", [256]),
        ("media", [512, 256]),
        ("profunda", [512, 256, 128]),
    ]
    betas = [1.0, 3.0]
    taxas = [1e-3]
    grade = []
    for nome_topo, camadas in topologias:
        for beta in betas:
            for taxa in taxas:
                grade.append({
                    "nome_topo": nome_topo,
                    "camadas": camadas,
                    "beta": beta,
                    "taxa": taxa,
                })
    return grade


def buscar_melhor_modelo(dim_latente, carregador_treino, carregador_val,
                         epocas_busca, epocas_final):
    """Faz a busca de hiperparametros para uma dimensao latente e re-treina a
    melhor configuracao por mais epocas.

    Retorna (modelo_final, melhor_config, tabela_resultados, historico_final)."""
    print(f"\n  >>> Buscando melhor topologia para espaco latente {dim_latente}D")
    tabela = []
    melhor = None

    for config in grade_hiperparametros():
        # cada configuracao parte da MESMA semente -> comparacao justa
        torch.manual_seed(SEED)
        modelo = VAE(camadas_ocultas=config["camadas"], dim_latente=dim_latente)
        resultado = treinar_vae(
            modelo, carregador_treino, carregador_val,
            epocas=epocas_busca, taxa_aprendizado=config["taxa"],
            beta=config["beta"], verboso=False,
        )
        linha = {**config, "val": resultado["melhor_val"]}
        tabela.append(linha)
        print(f"      topo={config['nome_topo']:9s} beta={config['beta']:.1f} "
              f"lr={config['taxa']:.0e} -> val={resultado['melhor_val']:.3f}")
        if melhor is None or resultado["melhor_val"] < melhor["val"]:
            melhor = linha

    print(f"  >>> Melhor config {dim_latente}D: topo={melhor['nome_topo']} "
          f"beta={melhor['beta']} lr={melhor['taxa']:.0e} (val={melhor['val']:.3f})")

    # Re-treina a MELHOR configuracao do zero, por mais epocas, para o modelo
    # final que sera usado na exploracao do espaco latente e no teste.
    print(f"  >>> Re-treinando a melhor config {dim_latente}D por ate {epocas_final} epocas")
    torch.manual_seed(SEED)
    modelo_final = VAE(camadas_ocultas=melhor["camadas"], dim_latente=dim_latente)
    resultado_final = treinar_vae(
        modelo_final, carregador_treino, carregador_val,
        epocas=epocas_final, taxa_aprendizado=melhor["taxa"],
        beta=melhor["beta"], verboso=True,
    )
    return resultado_final["modelo"], melhor, tabela, resultado_final["historico"]


# ==============================================================================
# 7. EXPLORACAO DO ESPACO LATENTE
# ==============================================================================
def extrair_latentes(modelo, x, lote=512):
    """Passa os dados pelo encoder e devolve as MEDIAS mu (a "posicao" de cada
    exemplo no espaco latente).

    POR QUE USAR mu (E NAO z AMOSTRADO)?
      Para visualizar/analisar o espaco latente queremos a posicao CENTRAL e
      deterministica de cada exemplo. z = mu + sigma*eps tem ruido; mu e a
      melhor estimativa da localizacao do exemplo. (Na GERACAO, ai sim
      amostramos.)"""
    modelo.eval()
    saidas = []
    with torch.no_grad():
        for inicio in range(0, x.size(0), lote):
            bloco = x[inicio:inicio + lote].to(DEVICE)
            mu, _ = modelo.codificar(bloco)
            saidas.append(mu.cpu().numpy())
    return np.concatenate(saidas, axis=0)


def _scatter_latente(ax, pontos_2d, rotulos, titulo):
    """Desenha um scatter 2D colorido por rotulo, com legenda das 10 classes."""
    for classe in range(10):
        mascara = rotulos == classe
        ax.scatter(pontos_2d[mascara, 0], pontos_2d[mascara, 1],
                   s=4, alpha=0.5, label=NOMES_CLASSES[classe])
    ax.set_title(titulo)
    ax.set_xlabel("dimensao 1")
    ax.set_ylabel("dimensao 2")
    ax.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.9)


def explorar_espaco_latente(modelo_2d, modelo_8d, x, y, pasta_saida):
    """Gera o grafico-chave do projeto: lado a lado, a representacao latente 2D
    (direta) e a projecao PCA do espaco latente 8D em 2D.

    Retorna a razao de variancia explicada pelos 2 primeiros componentes do PCA
    sobre o espaco 8D (para responder "quanto da variancia a projecao ilustra").
    """
    rotulos = y.numpy()

    # ---- (a) espaco latente 2D: ja e 2D, plota direto --------------------
    latentes_2d = extrair_latentes(modelo_2d, x)

    # ---- (b) espaco latente 8D -> PCA para 2D ----------------------------
    latentes_8d = extrair_latentes(modelo_8d, x)
    pca = PCA(n_components=2, random_state=SEED)
    proj_8d = pca.fit_transform(latentes_8d)
    var_explicada = pca.explained_variance_ratio_  # fracao por componente
    var_total_2 = float(var_explicada[:2].sum())

    # ---- figura lado a lado ----------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    _scatter_latente(ax1, latentes_2d, rotulos,
                     "Espaco latente 2D (VAE 2D) - colorido por rotulo")
    _scatter_latente(
        ax2, proj_8d, rotulos,
        f"Projecao PCA do espaco latente 8D -> 2D\n"
        f"variancia explicada: PC1={var_explicada[0]*100:.1f}%  "
        f"PC2={var_explicada[1]*100:.1f}%  (total={var_total_2*100:.1f}%)")
    fig.suptitle("Exploracao do espaco latente - Fashion-MNIST (conjunto de teste)",
                 fontsize=14)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "grafico_espaco_latente.png")
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    print(f"    salvo: {caminho}")
    return var_explicada


def gerar_amostras(modelo, pasta_saida, nome_arquivo, n_linhas=10, n_colunas=10):
    """GERACAO (prova de que e um modelo generativo): amostra z ~ N(0, I) e
    decodifica, formando uma grade de roupas NOVAS (nao sao copias do treino).

    Isso so produz imagens plausiveis porque o termo KL deixou o espaco latente
    proximo da Normal padrao - exatamente a regiao de onde amostramos aqui."""
    modelo.eval()
    with torch.no_grad():
        z = torch.randn(n_linhas * n_colunas, modelo.dim_latente, device=DEVICE)
        imagens = modelo.decodificar(z).cpu().numpy().reshape(-1, 28, 28)

    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(n_colunas, n_linhas))
    for i, ax in enumerate(eixos.flat):
        ax.imshow(imagens[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(f"Amostras geradas z~N(0,I) - VAE {modelo.dim_latente}D", y=1.01)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, nome_arquivo)
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    salvo: {caminho}")


def grade_manifold_2d(modelo, pasta_saida, n=20):
    """So faz sentido para o VAE 2D: varre uma grade regular no plano latente e
    decodifica cada ponto, revelando como as classes se organizam continuamente
    no espaco latente (o "mapa" do manifold). Usa a inversa da Normal para
    espacar os pontos de forma coerente com a densidade N(0,1)."""
    from scipy.stats import norm
    modelo.eval()
    grade_u = np.linspace(0.05, 0.95, n)
    valores = norm.ppf(grade_u)  # quantis da Normal padrao
    imagem_grande = np.zeros((28 * n, 28 * n), dtype=np.float32)
    with torch.no_grad():
        for i, yi in enumerate(valores):
            for j, xj in enumerate(valores):
                z = torch.tensor([[xj, yi]], dtype=torch.float32, device=DEVICE)
                img = modelo.decodificar(z).cpu().numpy().reshape(28, 28)
                imagem_grande[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(imagem_grande, cmap="gray")
    ax.axis("off")
    ax.set_title("Manifold do espaco latente 2D (grade decodificada)")
    caminho = os.path.join(pasta_saida, "grafico_manifold_2d.png")
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    salvo: {caminho}")


def plotar_reconstrucoes(modelo, x, pasta_saida, nome_arquivo, n=10):
    """Mostra n imagens originais (linha de cima) e suas reconstrucoes (linha de
    baixo) - evidencia a qualidade do termo de reconstrucao."""
    modelo.eval()
    with torch.no_grad():
        originais = x[:n].to(DEVICE)
        rec, _, _ = modelo(originais)
    originais = originais.cpu().numpy().reshape(-1, 28, 28)
    rec = rec.cpu().numpy().reshape(-1, 28, 28)

    fig, eixos = plt.subplots(2, n, figsize=(n, 2.4))
    for j in range(n):
        eixos[0, j].imshow(originais[j], cmap="gray"); eixos[0, j].axis("off")
        eixos[1, j].imshow(rec[j], cmap="gray"); eixos[1, j].axis("off")
    eixos[0, 0].set_ylabel("orig", rotation=0, labelpad=20)
    fig.suptitle(f"Reconstrucoes - VAE {modelo.dim_latente}D "
                 f"(cima: original | baixo: reconstruida)")
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, nome_arquivo)
    fig.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    salvo: {caminho}")


def plotar_curvas_aprendizado(hist_2d, hist_8d, pasta_saida):
    """Curvas de treino/validacao (loss total) para os dois modelos finais."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, hist, titulo in [(ax1, hist_2d, "VAE 2D"), (ax2, hist_8d, "VAE 8D")]:
        ax.plot(range(1, len(hist["treino"]) + 1), hist["treino"], label="treino")
        ax.plot(range(1, len(hist["val"]) + 1), hist["val"], label="validacao")
        ax.set_title(f"Curva de aprendizado - {titulo}")
        ax.set_xlabel("epoca"); ax.set_ylabel("loss (ELBO)")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    caminho = os.path.join(pasta_saida, "grafico_curvas_aprendizado_vae.png")
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    print(f"    salvo: {caminho}")


# ==============================================================================
# 8. PROGRAMA PRINCIPAL
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Projeto 3 - VAE generativo para o Fashion-MNIST.")
    parser.add_argument("--epocas", type=int, default=30,
                        help="epocas do re-treino final da melhor config (padrao: 30)")
    parser.add_argument("--epocas-busca", type=int, default=8,
                        help="epocas por configuracao na busca de hiperparametros (padrao: 8)")
    parser.add_argument("--lote", type=int, default=128, help="tamanho do lote (padrao: 128)")
    parser.add_argument("--pasta-dados", default="dados_fashion_mnist",
                        help="pasta de cache do dataset")
    parser.add_argument("--pasta-saida", default="RESULTS/project3",
                        help="pasta onde salvar os graficos")
    parser.add_argument("--amostra-latente", type=int, default=5000,
                        help="qtde de exemplos de teste usados nos graficos de latente "
                             "(subamostra para o scatter nao ficar pesado; padrao: 5000)")
    args = parser.parse_args()

    inicio = time.time()
    os.makedirs(args.pasta_saida, exist_ok=True)

    print("=" * 78)
    print("PROJETO 3 - VAE GENERATIVO PARA O FASHION-MNIST")
    print(f"Dispositivo: {DEVICE}  |  Semente: {SEED}")
    print("=" * 78)

    # ---- 1) dados ----------------------------------------------------------
    print("\n[1/5] Carregando Fashion-MNIST ...")
    x_treino_u, y_treino_u, x_teste_u, y_teste_u = carregar_fashion_mnist(args.pasta_dados)
    x_treino_full, y_treino_full = preparar_tensores(x_treino_u, y_treino_u)
    x_teste, y_teste = preparar_tensores(x_teste_u, y_teste_u)
    print(f"    treino+val: {x_treino_full.shape[0]} imagens | teste: {x_teste.shape[0]} imagens")

    # ---- 2) split treino/validacao SEM VAZAMENTO ---------------------------
    # Separamos a validacao ANTES de qualquer treino. O embaralhamento usa a
    # semente global, entao o split e reprodutivel. O TESTE (x_teste) nunca entra
    # na escolha de hiperparametros nem no early stopping.
    print("\n[2/5] Separando treino/validacao (90/10) - teste fica reservado ...")
    n = x_treino_full.shape[0]
    perm = np.random.permutation(n)           # usa np.random (semeado)
    n_val = int(0.10 * n)
    idx_val, idx_treino = perm[:n_val], perm[n_val:]
    x_treino, y_treino = x_treino_full[idx_treino], y_treino_full[idx_treino]
    x_val, y_val = x_treino_full[idx_val], y_treino_full[idx_val]
    print(f"    treino: {x_treino.shape[0]}  |  validacao: {x_val.shape[0]}  |  teste: {x_teste.shape[0]}")

    # Carregadores (DataLoader). So as entradas x importam (VAE e auto-supervisionado:
    # a "saida alvo" e a propria entrada). Os rotulos y guardamos a parte para os
    # graficos coloridos - eles NAO sao usados no treino.
    gerador = torch.Generator().manual_seed(SEED)  # embaralhamento reprodutivel
    carregador_treino = DataLoader(TensorDataset(x_treino), batch_size=args.lote,
                                   shuffle=True, generator=gerador)
    carregador_val = DataLoader(TensorDataset(x_val), batch_size=args.lote, shuffle=False)
    carregador_teste = DataLoader(TensorDataset(x_teste), batch_size=args.lote, shuffle=False)

    # ---- 3) busca de hiperparametros para 2D e 8D --------------------------
    print("\n[3/5] Otimizacao de hiperparametros (selecao pela loss de validacao) ...")
    modelo_2d, cfg_2d, tabela_2d, hist_2d = buscar_melhor_modelo(
        2, carregador_treino, carregador_val, args.epocas_busca, args.epocas)
    modelo_8d, cfg_8d, tabela_8d, hist_8d = buscar_melhor_modelo(
        8, carregador_treino, carregador_val, args.epocas_busca, args.epocas)

    # ---- 4) avaliacao final no TESTE (so agora!) ---------------------------
    print("\n[4/5] Avaliacao final no conjunto de TESTE (reservado) ...")
    teste_2d = avaliar(modelo_2d, carregador_teste, cfg_2d["beta"])
    teste_8d = avaliar(modelo_8d, carregador_teste, cfg_8d["beta"])
    print(f"    VAE 2D -> teste loss={teste_2d[0]:.3f} (recon={teste_2d[1]:.3f} kl={teste_2d[2]:.3f})")
    print(f"    VAE 8D -> teste loss={teste_8d[0]:.3f} (recon={teste_8d[1]:.3f} kl={teste_8d[2]:.3f})")

    # ---- 5) exploracao do espaco latente + geracao + graficos --------------
    print("\n[5/5] Explorando o espaco latente e gerando artefatos ...")
    # subamostra do teste para os scatters (desempenho; semeado)
    n_sub = min(args.amostra_latente, x_teste.shape[0])
    idx_sub = np.random.permutation(x_teste.shape[0])[:n_sub]
    x_sub, y_sub = x_teste[idx_sub], y_teste[idx_sub]

    var_explicada = explorar_espaco_latente(modelo_2d, modelo_8d, x_sub, y_sub, args.pasta_saida)
    plotar_curvas_aprendizado(hist_2d, hist_8d, args.pasta_saida)
    plotar_reconstrucoes(modelo_2d, x_teste, args.pasta_saida, "grafico_reconstrucoes_2d.png")
    plotar_reconstrucoes(modelo_8d, x_teste, args.pasta_saida, "grafico_reconstrucoes_8d.png")
    gerar_amostras(modelo_2d, args.pasta_saida, "grafico_amostras_geradas_2d.png")
    gerar_amostras(modelo_8d, args.pasta_saida, "grafico_amostras_geradas_8d.png")
    grade_manifold_2d(modelo_2d, args.pasta_saida)

    # ---- resumo final ------------------------------------------------------
    duracao = time.time() - inicio
    print("\n" + "=" * 78)
    print("RESUMO FINAL")
    print("=" * 78)
    print(f"Melhor config 2D: topologia={cfg_2d['camadas']} beta={cfg_2d['beta']} "
          f"lr={cfg_2d['taxa']:.0e} | val={cfg_2d['val']:.3f} | teste={teste_2d[0]:.3f}")
    print(f"Melhor config 8D: topologia={cfg_8d['camadas']} beta={cfg_8d['beta']} "
          f"lr={cfg_8d['taxa']:.0e} | val={cfg_8d['val']:.3f} | teste={teste_8d[0]:.3f}")
    print(f"PCA do latente 8D -> 2D: PC1={var_explicada[0]*100:.1f}%  "
          f"PC2={var_explicada[1]*100:.1f}%  total={var_explicada[:2].sum()*100:.1f}%")
    print(f"Artefatos salvos em: {args.pasta_saida}/")
    print(f"Tempo total: {duracao/60:.1f} min")
    print("=" * 78)


if __name__ == "__main__":
    main()
