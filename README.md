# Análise da Evolução da Área das Unidades de Conservação Costeiras com KDD

Este projeto aplica conceitos de KDD (Knowledge Discovery in Databases) e Mineração de Dados para analisar a evolução da área oficial das Unidades de Conservação (UCs) costeiras no Brasil, com base em dados históricos do CNUC.

O objetivo é identificar padrões de variação, estabilidade e comportamento temporal das UCs ao longo do período analisado, apoiando o projeto de extensão **“Unidades de Conservação é preciso”**.

---

## Objetivo

Investigar, com base em dados reais, hipóteses como:

- As UCs costeiras apresentam variação de área ao longo do tempo?
- A maior parte das UCs permanece estável em área oficial?
- A categoria de manejo está associada à direção da variação de área?
- UCs com séries temporais mais curtas tendem a apresentar maior instabilidade?
- UCs com maior cobertura temporal tendem a ser mais estáveis?

---

## Dados

### Fonte principal

- CNUC (Cadastro Nacional de Unidades de Conservação), anos de 2018 a 2025

### Estrutura dos dados

- Arquivos CSV anuais
- UCs filtradas para municípios costeiros
- Série temporal desbalanceada
- Nem todas as UCs aparecem em todos os anos

### Variáveis principais

- `codigo_uc`
- `nome_uc`
- `categoria_manejo`
- `grupo` (PI ou US)
- `area_ha`
- `ano`

---

## Pré-processamento

Os dados foram tratados em Python com as bibliotecas `pandas`, `numpy`, `scipy` e `scikit-learn`.

O pipeline realiza:

- leitura automática dos CSVs por pasta;
- padronização de colunas entre anos;
- conversão da área para formato numérico;
- normalização da categoria de manejo e do grupo;
- construção de painel histórico longitudinal;
- controle de qualidade para remover valores suspeitos.

### Variáveis derivadas

- `delta_area`: variação absoluta de área
- `delta_pct`: variação percentual
- `slope_ha_ano`: tendência temporal
- `slope_pct_ano`: tendência relativa
- `n_anos`: quantidade de anos observados
- `span_anos`: intervalo entre primeiro e último ano
- `coverage_ratio`: cobertura temporal da série
- `coorte`: UC nova ou existente
- `serie_tipo`: curta ou longa
- `qc_pass`: aprovação no controle de qualidade

### Controle de qualidade

Foram removidos casos com:

- áreas muito pequenas;
- variações extremas incompatíveis com interpretação analítica direta;
- valores ausentes ou inconsistentes.

---

## Principais Resultados

- A maioria das UCs costeiras apresenta estabilidade na área oficial registrada.
- Variações expressivas existem, mas ocorrem em uma parcela menor das unidades.
- A categoria de manejo tem peso relevante na análise.
- Séries mais curtas e casos com cobertura temporal diferente merecem interpretação cuidadosa.
- Os resultados refletem a área oficial registrada no CNUC, e não mudanças físicas diretas no território.

---

## Limitações

- Base administrativa, não ecológica direta
- Séries desbalanceadas
- UCs entram na base em anos diferentes
- Parte das variações pode refletir atualização cartográfica ou administrativa
- O estudo não mede diretamente erosão ou impacto ambiental físico

---

## Como Executar

### Instalar dependências

```bash
pip install pandas numpy scipy scikit-learn
```

## Como Executar

### Executar o script

```bash
python cnuc_analysis.py --folder csv
```
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