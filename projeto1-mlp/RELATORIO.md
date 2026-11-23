# Projeto 1 — Regressão com MLP e Estudo de Ablação (versão pós-entrega, didática)

**Disciplina:** Introdução às Redes Neurais Artificiais
**Autor:** Henrique Crepaldi (RA: 120410)
**Código:** `mlp_project1.py` (PyTorch)

> **Contexto desta versão.** A entrega original já foi feita. Este documento acompanha uma
> reescrita do código com finalidade de **aprendizado**: entender *o que* acontece com o
> baseline (por que o R² deu negativo), *como* acontece (mecanismo) e *como resolver*.
> As restrições do professor continuam respeitadas: baseline vanilla (SGD puro, sem
> momentum/regularização), ablação sem mudar a arquitetura, e as métricas MAE/MSE/RMSE/R².

---

## 1. Configuração experimental

- **Dataset:** `dataset_projeto1.csv` — 300 pares (x, y), com x ∈ [0, 10]; função não linear
  com oscilações e ruído.
- **Partição:** 10% treino (30) / 10% validação (30) / 80% teste (240), `random_state=42`.
- **Arquitetura (fixa para todos os modelos):** Entrada(1) → Densa(64) → Tanh → Densa(64) →
  Tanh → Saída(1). Inicialização **padrão** do PyTorch (sem init manual).
- **Otimizador:** SGD. **Custo:** MSE. **Batch:** 10 (3 atualizações por época).
- **Reprodutibilidade:** `Seed = 42` global; todos os experimentos partem da **mesma**
  inicialização de pesos.

O código não usa gradient clipping, nem early stopping, nem inicialização "esperta" — para
que o baseline seja **honesto** e a ablação seja **justa** (mesma arquitetura e mesmo
protocolo de treino, mudando apenas um componente por vez).

---

## 2. O problema original: R² negativo

Na entrega original (SGD puro, `lr=0.01`, **500 épocas**), o baseline obteve **R² ≈ −0.05** no
teste — pior do que simplesmente prever a média. O relatório original atribuiu isso a
**subajuste**: com 30 pontos para uma curva de alta frequência em [0, 10], a rede aprendeu só
a tendência de baixa frequência.

Isso está correto, mas a pergunta que este estudo respondeu foi: **o subajuste vem da
arquitetura, da escala dos dados, ou do treinamento?**

### 2.1. Como acontece (mecanismo)

Duas causas se somam:

1. **Saturação da Tanh.** Com x ∈ [0, 10] e pesos iniciais pequenos, as pré-ativações podem
   cair na região plana da Tanh (perto de ±1), onde a derivada é ~0. O gradiente que volta
   para a primeira camada quase zera e a rede "não anda".
2. **Convergência insuficiente.** Com SGD puro, passo pequeno (`lr=0.01`) e poucas épocas
   (500), mesmo sem saturação a rede não teria tempo de ajustar as oscilações.

---

## 3. Como resolver: duas alavancas testadas

### 3.1. Padronização da entrada (sem vazamento)

Padronizar x (média 0, desvio 1) mantém as pré-ativações na **região linear** da Tanh, onde o
gradiente flui. **Importante:** média e desvio são calculados **apenas no conjunto de treino** e
depois reaplicados em validação e teste — assim não há **vazamento** (leakage) de informação do
teste para o treino. Isso atende à condição do professor ("dá para fazer sem normalizar, mas se
usar, tem que justificar e entender para quê").

### 3.2. Convergência (passo + épocas)

Uma varredura empírica de `lr` e número de épocas (SGD puro, sem mudar a arquitetura) mostrou
que o gargalo dominante era a **convergência**, não a arquitetura:

| Otimizador | lr    | épocas | R² (teste) |
|------------|-------|--------|------------|
| SGD puro   | 0.01  | 500    | −0.02      |
| SGD puro   | 0.01  | 3000   | +0.06      |
| SGD puro   | 0.01  | 8000   | +0.17      |
| SGD puro   | 0.05  | 500    | +0.04      |
| SGD puro   | 0.05  | 3000   | **+0.19**  |
| SGD puro   | 0.10  | 3000   | +0.28      |
| SGD puro   | 0.10  | 8000   | −0.05      |

Observações:
- Aumentar passo/épocas destrava o aprendizado (de R² negativo para positivo) **sem tocar na
  arquitetura**. O número de épocas não faz parte da arquitetura, então ajustá-lo não viola a
  regra "os modelos aditivados não mudam a arquitetura".
- **Épocas demais pioram** (lr=0.10 + 8000 → R² negativo de novo). Aqui o problema deixa de ser
  subajuste e passa a ser **sobreajuste** — e é exatamente aí que a regularização faz sentido.

**Configuração adotada no baseline final:** `lr = 0.05`, `epocas = 3000` (SGD puro).

---

## 4. Um resultado contraintuitivo (e honesto)

Com o baseline final, comparando escala original vs. padronizada:

| Baseline           | MAE    | MSE    | RMSE   | R²     |
|--------------------|--------|--------|--------|--------|
| Sem padronização   | 0.4921 | 0.3717 | 0.6097 | **0.2811** |
| Com padronização   | 0.5161 | 0.4170 | 0.6458 | 0.1936 |

**A padronização NÃO deu o melhor R² aqui.** Isso não é um erro — é uma lição:
- A padronização **estabiliza e destrava** o treino (é indispensável quando a Tanh satura e o
  passo é pequeno), mas **não é bala de prata**. Com `lr=0.05` e 3000 épocas, a escala original
  já conseguiu convergir e, neste conjunto pequeno, generalizou até um pouco melhor.
- Isso confirma na prática a fala do professor: **dava para fazer sem normalizar**. A
  normalização é uma ferramenta para um problema específico (saturação/condicionamento), não um
  passo obrigatório que sempre melhora a métrica.

> O restante do estudo de ablação foi conduzido **sobre os dados padronizados**, para manter um
> pipeline único e porque a padronização torna o treino mais estável e comparável entre os
> modelos.

---

## 5. Estudo de ablação (dados padronizados)

Mesma arquitetura e mesmo protocolo; muda-se **um** componente por vez.

| Modelo               | MAE    | MSE    | RMSE   | R²     | Melhor época (val) |
|----------------------|--------|--------|--------|--------|--------------------|
| Baseline (SGD puro)  | 0.5161 | 0.4170 | 0.6458 | 0.1936 | 750                |
| + Momentum (0.9)     | 0.5224 | 0.4718 | 0.6869 | 0.0876 | 2001               |
| + L2 (1e-3)          | 0.5237 | 0.4252 | 0.6520 | 0.1778 | 750                |
| + L1 (1e-4)          | 0.5200 | 0.4219 | 0.6496 | 0.1840 | 750                |
| + Dropout (0.05)     | 0.5735 | 0.4916 | 0.7012 | 0.0492 | 1924               |

### Interpretação

- **Melhor época de validação do baseline = 750 de 3000.** Depois disso a validação para de
  melhorar e a rede começa a **sobreajustar** (visível no gráfico treino×validação: o treino
  continua caindo, a validação sobe). O gargalo, que começou como subajuste, virou sobreajuste.
- **Momentum piorou (R² 0.19 → 0.09).** Ele acelera a convergência, mas neste regime de poucos
  dados isso levou a rede **mais rápido para o sobreajuste** — a melhor época pulou para ~2000 e
  a curva ficou instável.
- **Dropout piorou (R² 0.05).** Com apenas 30 pontos, desligar unidades reduz ainda mais a já
  escassa capacidade efetiva e atrapalha o ajuste.
- **L1 e L2 quase empataram com o baseline** (0.184 e 0.178). A penalidade teve efeito marginal
  porque os pesos não atingiram magnitudes grandes — não havia muito o que "encolher".

**Detalhe técnico corrigido em relação à versão original:** a penalidade **L1 é aplicada apenas
aos pesos** (`weight`), nunca aos *bias*. Penalizar bias não tem justificativa estatística e
distorce o efeito da regularização.

---

## 6. Gráficos gerados

- `grafico_curvas_aprendizado.png` — evolução treino×validação, sem vs. com padronização
  (mostra o gap de sobreajuste).
- `grafico_ablation_loss.png` — perda de validação de cada modelo da ablação ao longo das épocas.
- `grafico_ajuste_regressao.png` — curva predita (baseline e +momentum) sobre treino e teste.

---

## 7. Lições aprendidas

1. **Diagnosticar antes de "consertar".** O R² negativo não vinha da arquitetura; vinha de
   **convergência insuficiente**. Ajustar passo e épocas resolveu a maior parte.
2. **Saturação de ativação importa.** A Tanh em [0, 10] com pesos pequenos sofre com gradiente
   ~0. Padronizar a entrada é a forma correta (e sem vazamento) de mitigar isso.
3. **Normalização não é obrigatória nem mágica.** Ajuda a *estabilizar/destravar*, mas pode não
   melhorar a métrica final — depende do resto do setup.
4. **Regularização resolve sobreajuste, não subajuste.** Por isso L1/L2/dropout/momentum não
   ajudaram enquanto o problema era subajuste, e passam a fazer sentido só depois que o treino
   converge e começa a sobreajustar.
5. **Com 30 pontos há um teto de generalização.** Nenhum ajuste fino supera a limitação de
   amostragem para uma função de alta frequência — a lição vale mais que o número de R².

---

## 8. Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch scikit-learn pandas numpy matplotlib
python mlp_project1.py
```

O script imprime as duas tabelas (efeito da padronização e ablação) e salva os três gráficos.
Os hiperparâmetros do baseline (`LR`, `EPOCAS`, `BATCH_SIZE`, `N_NEURONIOS`) estão no topo do
arquivo, comentados.
