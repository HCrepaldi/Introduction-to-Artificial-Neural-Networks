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
2. **Convergência insuficiente.** Com SGD puro, passo pequeno (`lr=0.01`) e **poucas épocas
   (500)**, mesmo sem saturação a rede não teria tempo de ajustar as oscilações. (Note: o
   problema original não era o `lr=0.01` em si, e sim o número baixo de épocas — ver seções 3.2
   e 3.3.)

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

Uma primeira correção usou `lr = 0.05`, que destravou o aprendizado. Porém, ao inspecionar a
curva de perda, notou-se um problema levantado em orientação: **a curva oscilava demais**
(serrilhada, com picos), sinal de learning rate ainda alto. Isso motivou a análise da seção 3.3.

### 3.3. O learning rate e a estabilidade do treino

Com passo grande (`lr=0.05`) e **batch pequeno (10 amostras)**, o gradiente estimado tem alta
variância; a cada atualização o SGD "passa do ponto" (*overshoot*) e a perda de validação fica
**quicando** em torno do mínimo, em vez de descer suave. Para quantificar isso, mediu-se o
*jitter* — o desvio-padrão das variações de perda de época para época na segunda metade do
treino (quanto maior, mais serrilhada é a curva):

| lr     | R² (teste) | Jitter (variação) | Melhor época (val) |
|--------|------------|-------------------|--------------------|
| 0.05   | 0.19       | 0.0297 (instável) | 750                |
| 0.02   | 0.13       | 0.0302            | 750                |
| **0.01** | **0.06**  | **0.0169 (~45% menor)** | 1179          |
| 0.005  | 0.05       | 0.0047 (bem suave)| 2472               |
| 0.002  | 0.004      | 0.0007 (super liso)| 2980              |

Há um **trade-off** claro: abaixar o `lr` reduz drasticamente a instabilidade, mas passos muito
pequenos deixam o SGD puro lento demais para escapar de mínimos rasos em 3000 épocas, derrubando
o R². O `lr=0.05` só alcançava R² alto **às custas** de um treino instável — ou seja, o "bom"
resultado dependia de onde o quicar parava, não de convergência confiável.

**Configuração adotada no baseline final:** `lr = 0.01`, `epocas = 3000` (SGD puro). É o ponto
de equilíbrio: curva visivelmente mais estável (jitter ~45% menor que com `lr=0.05`) e
convergência saudável.

---

## 4. Um resultado contraintuitivo (e honesto)

Com o baseline final (`lr=0.01`), comparando escala original vs. padronizada:

| Baseline           | MAE    | MSE    | RMSE   | R²     |
|--------------------|--------|--------|--------|--------|
| Sem padronização   | 0.5784 | 0.5124 | 0.7158 | 0.0091 |
| Com padronização   | 0.5570 | 0.4841 | 0.6958 | **0.0637** |

Com o passo estável, a **padronização passa a ajudar** (R² 0.009 → 0.064): sem ela, a saturação
da Tanh combinada ao passo pequeno trava o aprendizado. Vale registrar uma lição observada
durante o estudo: com `lr=0.05` (instável), a escala original chegava a dar R² maior — mas de
forma **não confiável**, dependente do ponto onde o treino oscilante parava. Ou seja:

- A padronização **estabiliza e destrava** o treino, especialmente quando a Tanh satura e o
  passo é pequeno; **não é, porém, um passo mágico** que sempre maximiza a métrica.
- Isso é coerente com a fala do professor: **dava para fazer sem normalizar**. A normalização é
  ferramenta para um problema específico (saturação/condicionamento numérico), não obrigação.

> O restante do estudo de ablação foi conduzido **sobre os dados padronizados**, para manter um
> pipeline único, estável e comparável entre os modelos.

---

## 5. Estudo de ablação (dados padronizados)

Mesma arquitetura e mesmo protocolo; muda-se **um** componente por vez.

| Modelo               | MAE    | MSE    | RMSE   | R²     | Melhor época (val) |
|----------------------|--------|--------|--------|--------|--------------------|
| Baseline (SGD puro)  | 0.5570 | 0.4841 | 0.6958 | 0.0637 | 1179               |
| + Momentum (0.9)     | 0.4861 | 0.4338 | 0.6586 | **0.1611** | 2964           |
| + L2 (1e-3)          | 0.5585 | 0.4821 | 0.6943 | 0.0677 | 1179               |
| + L1 (1e-4)          | 0.5569 | 0.4821 | 0.6943 | 0.0677 | 1179               |
| + Dropout (0.05)     | 0.5780 | 0.5010 | 0.7078 | 0.0311 | 2153               |

### Interpretação

- **Momentum foi o melhor modelo (R² 0.06 → 0.16).** Este é um resultado dependente do learning
  rate e vale destacar: com `lr=0.05` (instável, versão anterior deste estudo) o momentum
  *piorava*, porque acelerava o *overshoot* e a curva quicava. Com o `lr=0.01` estável, o
  momentum faz o que se espera dele — **acelera a convergência de forma útil**, ajudando o SGD
  puro (que sozinho é lento com passo pequeno) a descer de forma suave e consistente. **Lição: o
  efeito do momentum depende do learning rate.**
- **Melhor época de validação do baseline = 1179 de 3000.** Depois disso a validação para de
  melhorar e volta a subir (visível no gráfico treino×validação: o treino continua caindo, a
  validação faz um "U"). O gargalo, que começou como subajuste, torna-se sobreajuste.
- **Dropout piorou (R² 0.03).** Com apenas 30 pontos, desligar unidades reduz ainda mais a já
  escassa capacidade efetiva e atrapalha o ajuste.
- **L1 e L2 praticamente empataram com o baseline** (ambos 0.068 vs. 0.064). A penalidade teve
  efeito marginal porque os pesos não atingiram magnitudes grandes — não havia muito o que
  "encolher".

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
   **convergência insuficiente** (poucas épocas). Ajustar passo e épocas resolveu a maior parte.
2. **Learning rate controla a estabilidade.** Passo grande (`lr=0.05`) com batch pequeno faz a
   perda oscilar/quicar (*overshoot*); um bom resultado assim é pouco confiável. Baixar para
   `lr=0.01` reduziu o *jitter* em ~45% e deu um treino estável — porém `lr` pequeno demais torna
   o SGD puro lento. Existe um ponto de equilíbrio.
3. **Saturação de ativação importa.** A Tanh em [0, 10] com pesos pequenos sofre com gradiente
   ~0. Padronizar a entrada (sem vazamento) é a forma correta de mitigar isso.
4. **Normalização não é obrigatória nem mágica.** Ajuda a *estabilizar/destravar*, mas seu ganho
   depende do resto do setup (passo, épocas).
5. **O efeito do momentum depende do learning rate.** Com passo instável ele piorava (acelerava
   o overshoot); com passo estável ele virou o melhor modelo (acelerou a convergência útil).
6. **Regularização resolve sobreajuste, não subajuste.** Por isso L1/L2/dropout não ajudaram
   enquanto o problema era subajuste, e só passam a fazer sentido depois que o treino converge e
   começa a sobreajustar.
7. **Com 30 pontos há um teto de generalização.** Nenhum ajuste fino supera a limitação de
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
