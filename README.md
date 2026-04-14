# Análise de Vulnerabilidade Costeira com KDD

Este projeto aplica conceitos de KDD (Knowledge Discovery in Databases) e Mineração de Dados para analisar a relação entre vulnerabilidade costeira e a presença de Unidades de Conservação (UC) no Brasil.

Além disso, incorpora uma análise temporal das UCs com base em dados do CNUC (2018–2025), permitindo avaliar mudanças reais nas áreas protegidas ao longo do tempo.

O objetivo é transformar dados em conhecimento útil, apoiando decisões no contexto do projeto de extensão **"Unidades de Conservação é preciso"**.

---

## Objetivo

Investigar, com base em dados, hipóteses como:

* A presença de Unidades de Conservação reduz a vulnerabilidade costeira?
* O tipo de UC influencia essa vulnerabilidade?
* A cobertura de restinga explica melhor a vulnerabilidade do que apenas a existência de UC?
* A urbanização aumenta a vulnerabilidade?
* As UCs estão realmente expandindo ou permanecem estáveis ao longo do tempo?
* Existem padrões estruturais de crescimento ou redução nas áreas protegidas?

---

## Abordagem Metodológica

O projeto segue o processo completo de KDD:

1. Seleção e integração de dados
2. Limpeza e padronização
3. Transformação (features derivadas)
4. Mineração de dados
5. Interpretação dos resultados

---

## Dados

### Fonte principal

* CNUC (Cadastro Nacional de Unidades de Conservação 2018-2025)

### Estrutura dos dados de UCs

* Série temporal anual (2018–2025)
* Apenas municípios costeiros
* UCs podem surgir ao longo do tempo (dataset dinâmico)
* Nem todas as UCs existem em todos os anos

### Variáveis principais

* `codigo_uc`
* `nome_uc`
* `categoria_manejo`
* `grupo` (PI ou US)
* `area_ha`
* `ano`

---

## Engenharia de Features

O pipeline gera variáveis derivadas importantes:

* `delta_pct`: variação percentual da área
* `delta_area`: variação absoluta
* `slope_ha_ano`: tendência temporal (regressão)
* `slope_pct_ano`: tendência relativa
* `n_anos`: tamanho da série
* `serie_tipo`: curta ou longa
* `coorte`: UC nova ou antiga
* `qc_pass`: controle de qualidade

### Controle de Qualidade (QC)

Filtragem de dados potencialmente inconsistentes:

* Áreas muito pequenas (< 5 ha)
* Variações extremas (> 500%)
* Valores inválidos ou ausentes

---

## Técnicas Utilizadas

### 1. Análise Exploratória (EDA)

* Estatísticas descritivas
* Distribuição de variação de área
* Comparações por:

  * categoria de manejo
  * grupo (PI vs US)
  * coorte
  * tipo de série

---

### 2. Classificação (Aprendizado Supervisionado)

* Árvore de decisão (estilo J48 / C4.5)
* Validação cruzada estratificada
* Métrica principal: F1-score macro

**Resultado atual:**

* F1 macro ≈ 0.36–0.37
* Forte desbalanceamento de classes (predominância de "estável")

---

### 3. Agrupamento (Aprendizado Não Supervisionado)

* K-Means (k=3)
* Features:

  * delta_pct
  * área inicial/final (log)
  * tendência temporal

**Padrões identificados:**

* Cluster dominante: UCs estáveis (~99%)
* Pequenos clusters de expansão real

---

### 4. Regras de Associação

* Extração simplificada de padrões
* Métricas:

  * suporte
  * confiança
  * lift

**Exemplo de padrão:**

* UCs de uso sustentável → leve tendência maior de redução (fraca evidência)

---

## Principais Resultados

* A maioria das UCs é **extremamente estável em área**
* Pequenas variações podem refletir:

  * ajustes cartográficos
  * atualização de dados
* Expansões reais existem, mas são raras
* Reduções também são raras, porém relevantes
* O modelo preditivo tem baixa capacidade de generalização (esperado)

---

## Limitações

* Dados administrativos (não ecológicos diretos)
* Mudanças podem ser artefatos de medição
* Forte desbalanceamento de classes
* Entrada e saída de UCs ao longo do tempo (dataset não estacionário)
* Ausência de variáveis causais (políticas públicas, fiscalização, etc.)

---

## Validação e Confiabilidade

Medidas adotadas para evitar "alucinações" analíticas:

* Controle de qualidade (QC)
* Uso de mediana (robusto a outliers)
* Regressão temporal por UC
* Validação cruzada estratificada
* Interpretação cautelosa (sem inferência causal indevida)

---

## Como Executar

### 1. Instalar dependências

```bash
pip install pandas numpy scipy scikit-learn
```

### 2. Executar o script

```bash
python cnuc_analysis.py --folder csv
```

---

## Estrutura Esperada

```
/csv
  cnuc_2018.csv
  cnuc_2019.csv
  ...
  cnuc_2025.csv
```

---

## Saídas

O script gera:

* Estatísticas no terminal
* Modelos interpretáveis (árvore de decisão)
* Clusters de comportamento
* Regras de associação

(Sugestão futura: exportação automática para `/results`)

---

## Aplicabilidade

Este projeto pode ser utilizado:

* Em contexto acadêmico (TCC, artigos, disciplinas)
* Como base exploratória para políticas públicas
* Como apoio a estudos ambientais

**Não recomendado como ferramenta decisória isolada**, devido às limitações dos dados.

---

## Contexto Acadêmico

* Disciplina: Inteligência Artificial II
* Tema: KDD e Mineração de Dados
* Projeto de Extensão: *Unidades de Conservação é preciso*

---

## Autores

* Cassiano de Sena Crispim
* Davi Negreiros Carneiro Rangel
* Eduardo da Rocha Weber
* Erik Luiz Gervasi
* Leonardo Alberto da Silva
* Victor André Uller

---

## Licença

Uso acadêmico.
