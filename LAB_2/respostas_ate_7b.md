# Lab 2 IAAut: o que dizer até à R7.b)

Paciente 4: 70 sequências → 56 de treino e 14 de teste (as últimas 20%, pela ordem original). Classes E1–E5 → 0–4.
Teste: E1=2, E2=5, E3=2, E4=1, E5=4. **Com tão poucas amostras, 1 erro numa classe com 1 ou 2 amostras faz cair a bacc 0,1 ou 0,2.**

---

## R3 (implementação)
- **3b**: filtro `Patient_ID == 4`, `mapa` E1..E5 → 0..4, últimas `round(0.2·70) = 14` para teste.
- **3c**: `construir_X` aplica `extrair_atributos` a cada sequência → Xtrain (56×268), Xtest (14×268). Cada sequência passa a ser 1 vetor de 268 features (médias, desvios-padrão de velocidades, amplitudes de ângulos…).
- **3d**: `confusion_matrix` e `balanced_accuracy_score` do sklearn. A bacc é a média do recall de cada classe, por isso não é enganada pelo desequilíbrio (E2 tem 5 no teste, E4 só 1).

## R4: Árvores de decisão
| Profundidade | Treino | Teste |
|---|---|---|
| 1 | 0,389 | 0,400 |
| 2 | 0,749 | 0,760 |
| 3 | 0,967 | **0,910** |
| 4 | 1,000 | 0,910 |
| 5 | 1,000 | 0,910 |

- **4a/4b**: com prof. 1–2 há *underfitting* (só 2–4 folhas, não chegam para 5 classes). A partir da 3 o teste estabiliza em 0,91: a prof. 4/5 decora o treino (1,000) mas não melhora o teste, ou seja, começa a haver *overfitting*. **Melhor árvore: prof. 3** (mais simples com o mesmo teste).
- Erros no teste (prof. 3/4): 1 E2 → E1 e 1 E5 → E3. E1/E2 são os exercícios mais parecidos.
- **4c** (prof. 4): a raiz é `cotovelo_esquerdo_desvio_padrao_vel_y`. Separa os exercícios de braço com muito movimento vertical (E3, E4, E5) dos mais parados (E1, E2). Depois o joelho e o cotovelo separam E1 de E2; o nariz isola E4 e o punho direito separa E3 de E5. Usa só velocidades e amplitudes, o que faz sentido fisicamente e se interpreta bem.
- ⚠️ Corrigi uma coisa: a célula da 4c tinha `arvore_4 = arvores[3]` (imprimia a de prof. 3). Agora está `arvores[4]`.

## R5.a): Naive Bayes com features manuais (FEITO AGORA)
`GaussianNB` treinado em Xtrain/Xtest sem normalizar (o NB gaussiano estima média e variância por feature e por classe, por isso a escala não importa).

| | Treino | Teste |
|---|---|---|
| NB manuais | 0,911 | 0,900 |
| Árvore prof. 3 | 0,967 | 0,910 |

- **Matriz**: tudo certo menos o E1. No treino há 7 E1 → E2 (e 1 E1 → E3), e no teste 1 dos 2 E1 → E2. E2, E3, E4 e E5 estão 100% certos.
- **Comparação com a árvore**: no teste ficam praticamente empatados (0,90 contra 0,91). Erram de forma diferente: a árvore confunde E2 → E1 e E5 → E3, enquanto o NB só confunde E1 → E2. Com 14 sequências de teste, 0,01 é menos de um erro, por isso não dá para dizer que um é melhor (e a prof. 3 foi escolhida a olhar para o teste). O treino do NB mais baixo (0,911, 7 dos 18 E1 errados) é **viés** do modelo, não "pouco overfitting".
- **Porque o NB falha no E1**: assume as 268 features independentes, dada a classe, o que é falso (os keypoints vizinhos estão muito correlacionados, por exemplo ombro/cotovelo/punho). O NB "conta várias vezes" a mesma informação. Como E1 e E2 são parecidos, essa contagem dupla puxa o E1 para o E2. Além disso há 268 features e só ~11 sequências por classe para estimar 2 parâmetros por feature. (É uma hipótese que não testámos: por exemplo, NB com menos features ou com PCA.)
- Coerente com a 6: o BoP com K=5 dá o mesmo 0,900 no teste com o mesmo erro (1 E1 → E2).

## R6: Bag-of-Poses + NB
| K | Treino | Teste |
|---|---|---|
| 2 | 0,400 | 0,400 |
| 5 | 0,811 | **0,900** |
| 10 | 0,811 | 0,800 |
| 15 | 0,911 | 0,800 |

- K=2 tem poses a menos: quase tudo vai parar ao mesmo centróide → prevê quase tudo E1.
- Ao aumentar K o treino sobe, mas o teste desce a partir de K=5: os histogramas ficam com mais dimensões (mais parâmetros para o NB estimar com 56 sequências) e mais esparsos → overfitting.
- **6b**: escolhemos K=5 (melhor teste). Pelos histogramas médios, E1 e E2 caem quase todo no mesmo centróide (0,98 e 1,00) e por isso são indistinguíveis. É aí que estão os erros. E5 tem um centróide só seu.
- Comparação: BoP K=5 (0,900) ≈ NB manuais (0,900) ≈ árvore prof. 3 (0,910). O BoP usa só 5 features, contra 268.

## R7: Regressão logística (SGD)
Implementação (`rl.py`): pesos W (p×5) e offset b (5) a zero; em cada época percorre as amostras **sempre pela mesma ordem** e, para cada uma, faz softmax(Wᵀx + b), calcula o erro = prob − one-hot e atualiza W ← W − η·x·erroᵀ, b ← b − η·erro. No fim da época guarda a entropia cruzada total. As features são normalizadas com StandardScaler (ajustado só no treino).

### R7.a) (lr = 0,01; 50 épocas)
| | Treino | Teste |
|---|---|---|
| Manuais | 1,000 | 0,760 |
| BoP K=5 | 0,789 | 0,800 |

- Manuais: 268 features e 56 amostras → os dados são linearmente separáveis e o modelo decora o treino: **overfitting** (teste 0,76, pior que o NB e a árvore). No teste erra 1 E2 → E1 e os 2 E3 → E1.
- BoP: E1 e E2 têm histogramas quase iguais, por isso um modelo linear não os consegue separar e põe todos os E2 em E1 (há mais E1 no treino: 18 contra 15). As outras classes estão 100% certas.

### R7.b) (lr ∈ {0,01; 0,001; 0,0001; 0,00001}) (FEITO AGORA)
Código: para cada conjunto de features e cada lr treina um modelo, junta as 4 curvas de custo num gráfico (épocas × entropia cruzada) e faz uma grelha 2×4 de matrizes de confusão (treino em cima, teste em baixo).

| lr | Manuais treino | Manuais teste | BoP treino | BoP teste |
|---|---|---|---|---|
| 0,01 | 1,000 | 0,760 | 0,789 | 0,800 |
| 0,001 | 1,000 | 0,720 | 0,789 | 0,800 |
| 0,0001 | 0,937 | 0,740 | 0,789 | 0,800 |
| 0,00001 | 0,847 | 0,650 | 0,789 | 0,800 |

**Curvas de custo:**
- Todas começam perto de **56·ln 5 ≈ 90,1**: com os pesos a zero, o softmax dá 1/5 a cada classe, logo o custo inicial é N·ln K. Bom ponto para dizer à stora.
- Quanto maior o lr, mais rápido desce. Manuais: lr 0,01 chega a ~0,7 (quase zero, porque os dados são separáveis), 0,001 a ~7, 0,0001 a ~35, e 1e-5 quase não sai do sítio (~72). Nenhuma diverge: 0,01 é estável para dados normalizados.
- BoP: a 0,01 estabiliza em ~25 e não chega a zero, porque o E1 e o E2 não são separáveis e o custo tem um patamar. Com lr pequenos, 50 épocas não chegam para convergir (1e-5 fica em ~89).

**Matrizes de confusão:**
- Manuais: com lr pequeno o modelo **ainda não convergiu** em 50 épocas (não é falta de capacidade; o treino desce para 0,937 e 0,847) e as previsões puxam para a classe com mais amostras (E1). Com 1e-5 os 5 E2 de teste vão todos para E1. Com lr grande decora o treino. O melhor teste é lr 0,01 (0,76), mas fica sempre abaixo do NB e da árvore.
- BoP: **as matrizes são iguais para todos os lr.** Explicação: o E1 e o E2 caem quase todo no mesmo centróide (0,98 e 1,00), por isso depois de normalizados ficam no mesmo ponto e nenhuma fronteira linear os separa, seja qual for o lr. Decide o offset, que favorece o E1 (18 contra 15 no treino). As outras três classes separam-se logo nas primeiras épocas. O lr só muda a confiança (o custo). O limite é a representação, não o treino.
- **O que não temos:** só traçámos o custo de treino. Para provar overfitting faltava a curva do custo no teste. Os gráficos usam eixo y logarítmico e o custo é a soma nas 56 amostras.

**Conclusão até aqui:** para o paciente 4, os melhores modelos no teste são a árvore de prof. 3 (0,91), o NB com manuais e o NB com BoP K=5 (0,90). A regressão logística com manuais sofre de overfitting (268 features para 56 amostras). Com BoP não consegue separar E1/E2. O erro comum a todos é a confusão E1 ↔ E2.
