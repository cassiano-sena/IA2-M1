# Análise de Vulnerabilidade Costeira com KDD

Este projeto aplica conceitos de KDD (Knowledge Discovery in Databases) e Mineração de Dados para analisar a relação entre vulnerabilidade costeira e a presença de Unidades de Conservação (UC) no Brasil.

O objetivo é transformar dados em conhecimento útil, apoiando decisões no contexto do projeto de extensão "Unidades de Conservação é preciso".

---

## Objetivo

Investigar, com base em dados, hipóteses como:

- A presença de Unidades de Conservação reduz a vulnerabilidade costeira?
- O tipo de UC influencia essa vulnerabilidade?
- A cobertura de restinga explica melhor a vulnerabilidade do que apenas a existência de UC?
- A urbanização aumenta a vulnerabilidade?

---

## Técnicas Utilizadas

O projeto implementa etapas completas do processo de KDD:

### 1. Análise Exploratória (EDA)
- Distribuições
- Correlações (Spearman)
- Testes estatísticos (Mann-Whitney, Kruskal-Wallis)
- Regressão linear

### 2. Classificação (Aprendizado Supervisionado)
- Árvore de decisão (estilo J48 / C4.5)
- Validação cruzada (10-fold)
- Métricas: acurácia, matriz de confusão, F1-score

### 3. Agrupamento (Aprendizado Não Supervisionado)
- K-Means (k=3)
- Identificação de perfis de municípios

### 4. Regras de Associação
- Extração de padrões relevantes
- Implementação simplificada (sem Apriori completo)

---

## Como Executar

### 1. Instalar dependências

```bash
pip install pandas numpy scipy scikit-learn
````
### 2. Executar o script
````bash
python3 cnuc_analysis_from_log.py
````
Ou especificando arquivos:
````bash
python3 cnuc_analysis_from_log.py --csv municipios_costeiros_vulnerabilidade.csv
````
CSV extra opcional:
````bash
python3 cnuc_analysis_from_log.py --extra-csv outro_arquivo.csv
````
## Dados

O dataset principal contém municípios costeiros brasileiros com atributos como:

- vulnerabilidade_ivc
- vulnerabilidade_classe
- tem_uc
- tipo_uc
- restinga_pct
- costa_uc_km
- urbanizacao_pct
- erosao_ativa

Os dados foram adaptados de bases reais de fontes como MMA, ICMBio, sendo as principais o IBGE BC250 2025 e o CNUC.

## Principais Resultados
- A urbanização apresenta forte correlação positiva com a vulnerabilidade.
- A cobertura de restinga apresenta forte correlação negativa com a vulnerabilidade.
- A presença de UC, isoladamente, não mostrou impacto estatisticamente significativo.
- A restinga se mostrou um fator mais explicativo que a existência de UC.
Contexto Acadêmico

### Disciplina: Inteligência Artificial II
### Tema: KDD e Mineração de Dados
### Projeto de Extensão: Unidades de Conservação é preciso

### Autores
- CASSIANO DE SENA CRISPIM
- DAVI NEGREIROS CARNEIRO RANGEL
- EDUARDO DA ROCHA WEBER
- ERIK LUIZ GERVASI
- LEONARDO ALBERTO DA SILVA
- VICTOR ANDRÉ ULLER

### Licença

Uso acadêmico.