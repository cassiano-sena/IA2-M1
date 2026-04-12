"""
Análise do CNUC a partir do CSV municipios_costeiros_vulnerabilidade.csv.

Fluxo principal:
1) Lê o CSV municipios_costeiros_vulnerabilidade.csv
2) Executa análises exploratórias
3) Treina uma árvore de decisão estilo J48/WEKA
4) Executa K-Means e regras de associação simplificadas

Uso:
    ./cnuc_analysis_from_log.py
    ./cnuc_analysis_from_log.py --csv municipios_costeiros_vulnerabilidade.csv
    ./cnuc_analysis_from_log.py --csv municipios_costeiros_vulnerabilidade.csv --extra-csv bc250_2025-hid_massa_dagua_a.csv

Observação:
- O script procura os arquivos primeiro na mesma pasta do script.
- O arquivo extra é opcional e não interfere no fluxo principal.
"""
"""
Para executar em uma máquina local, salve o arquivo como `cnuc_analysis_from_log.py`, coloque os CSVs na mesma pasta e rode:

```bash
chmod +x cnuc_analysis_from_log.py
./cnuc_analysis_from_log.py
```

Se estiver no Windows, rode com:

```bash
python cnuc_analysis_from_log.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def script_dir() -> Path:
    """Diretório onde este script está salvo."""
    return Path(__file__).resolve().parent


def resolve_file(path_arg: str | None, default_name: str) -> Path:
    """
    Resolve um arquivo com prioridade para:
    1) caminho informado pelo usuário
    2) arquivo no mesmo diretório do script
    3) arquivo no diretório atual de execução
    """
    candidates = []

    if path_arg:
        candidates.append(Path(path_arg).expanduser())

    candidates.append(script_dir() / default_name)
    candidates.append(Path.cwd() / default_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Não encontrei o arquivo '{default_name}'. Caminhos tentados:\n{tried}"
    )


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Erro ao ler CSV '{path}': {e}") from e


def print_header(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def exploratory_analysis(df: pd.DataFrame) -> None:
    print_header("ANÁLISE EXPLORATÓRIA")

    required = [
        "vulnerabilidade_ivc",
        "tem_uc",
        "tipo_uc",
        "restinga_pct",
        "costa_uc_km",
        "urbanizacao_pct",
        "erosao_ativa",
        "vulnerabilidade_classe",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV principal sem colunas esperadas: {missing}")

    print("\n--- 1. CORRELAÇÕES (Spearman) ---")
    for v in ["restinga_pct", "costa_uc_km", "urbanizacao_pct", "erosao_ativa"]:
        r, p = stats.spearmanr(df["vulnerabilidade_ivc"], df[v])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"vuln × {v:20s}: r={r:+.3f} p={p:.4f} {sig}")

    print("\n--- 2. MANN-WHITNEY: IVC com UC vs. sem UC ---")
    com_uc = df.loc[df["tem_uc"] == "Sim", "vulnerabilidade_ivc"]
    sem_uc = df.loc[df["tem_uc"] == "Nao", "vulnerabilidade_ivc"]
    stat, p = stats.mannwhitneyu(com_uc, sem_uc, alternative="less")
    print(f"Média IVC com UC : {com_uc.mean():.2f}")
    print(f"Média IVC sem UC : {sem_uc.mean():.2f}")
    print(f"U={stat:.0f} p={p:.4f} {'✓ sig.' if p < 0.05 else '✗ não sig.'}")

    print("\n--- 3. KRUSKAL-WALLIS: IVC por tipo de UC ---")
    grupos = [df.loc[df["tipo_uc"] == t, "vulnerabilidade_ivc"].values for t in df["tipo_uc"].unique()]
    stat, p = stats.kruskal(*grupos)
    print(f"H={stat:.3f} p={p:.4f} {'✓ sig.' if p < 0.05 else '✗ não sig.'}")
    print("Média IVC por tipo:")
    print(df.groupby("tipo_uc")["vulnerabilidade_ivc"].mean().round(2).sort_values().to_string())

    print("\n--- 4. REGRESSÃO: restinga_pct → vulnerabilidade_ivc ---")
    slope, intercept, r, p, se = stats.linregress(df["restinga_pct"], df["vulnerabilidade_ivc"])
    print(f"R²={r**2:.3f} slope={slope:.4f} p={p:.4f}")
    print(f"Interpretação: cada 10pp a mais de restinga → {slope * 10:.2f} de mudança no IVC")

    print("\n--- 5. EROSÃO ATIVA × PRESENÇA DE UC ---")
    ct = pd.crosstab(df["tem_uc"], df["erosao_ativa"], normalize="index").round(3) * 100
    ct.columns = ["Sem erosão", "Erosão mod.", "Erosão severa"]
    print(ct.to_string())

    print("\n--- 6. PREVIEW DOS DADOS PARA WEKA ---")
    cols_preview = [
        "tem_uc",
        "tipo_uc",
        "restinga_pct",
        "costa_uc_km",
        "urbanizacao_pct",
        "erosao_ativa",
        "vulnerabilidade_classe",
    ]
    print(df[cols_preview].head(10).to_string(index=False))


def j48_like_tree(df: pd.DataFrame) -> None:
    print_header("ÁRVORE DE DECISÃO ESTILO J48")

    le_uc = LabelEncoder()
    le_tipo = LabelEncoder()

    X = pd.DataFrame(
        {
            "tem_uc": le_uc.fit_transform(df["tem_uc"]),
            "tipo_uc": le_tipo.fit_transform(df["tipo_uc"]),
            "restinga_pct": df["restinga_pct"],
            "costa_uc_km": df["costa_uc_km"],
            "urbanizacao_pct": df["urbanizacao_pct"],
            "erosao_ativa": df["erosao_ativa"],
        }
    )
    y = (df["vulnerabilidade_classe"] == "Alta").astype(int)

    clf = DecisionTreeClassifier(
        criterion="entropy",
        min_samples_leaf=2,
        ccp_alpha=0.01,
        random_state=SEED,
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    print("=== CROSS-VALIDATION 10-FOLD (equiv. WEKA) ===")
    print(f"Acurácia média: {scores.mean() * 100:.1f}% (±{scores.std() * 100:.1f}%)")

    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = (y_pred == y).mean()

    print("\n=== TREINAMENTO NO CONJUNTO COMPLETO ===")
    print(f"Acurácia: {acc * 100:.1f}%")
    print(f"Instâncias classificadas corretamente: {(y_pred == y).sum()} de {len(y)}")

    print("\n=== MATRIZ DE CONFUSÃO ===")
    cm = confusion_matrix(y, y_pred)
    print("              Pred. Baixa  Pred. Alta")
    print(f"Real Baixa        {cm[0,0]:3d}        {cm[0,1]:3d}")
    print(f"Real Alta         {cm[1,0]:3d}        {cm[1,1]:3d}")

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y, y_pred, target_names=["Baixa", "Alta"]))

    print("\n=== ÁRVORE DE DECISÃO (texto) ===")
    tree_rules = export_text(clf, feature_names=list(X.columns))
    print(tree_rules[:2000])

    print("\n=== IMPORTÂNCIA DOS ATRIBUTOS ===")
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"{feat:20s}: {imp:.3f} {bar}")


def clustering_and_rules(df: pd.DataFrame) -> None:
    print_header("AGRUPAMENTO K-MEANS E REGRAS DE ASSOCIAÇÃO")

    print("=== AGRUPAMENTO K-MEANS (k=3) ===")
    features_cluster = ["vulnerabilidade_ivc", "restinga_pct", "urbanizacao_pct", "erosao_ativa"]
    X = df[features_cluster].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    df2 = df.copy()
    df2["cluster"] = kmeans.fit_predict(X_scaled)

    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c]
        print(f"\nCluster {c} ({len(sub)} municípios):")
        print(f" IVC médio: {sub['vulnerabilidade_ivc'].mean():.2f}")
        print(f" Restinga média: {sub['restinga_pct'].mean():.1f}%")
        print(f" Urbanização: {sub['urbanizacao_pct'].mean():.1f}%")
        print(f" Com UC: {(sub['tem_uc'] == 'Sim').mean() * 100:.0f}%")
        print(f" Regiões: {', '.join(sub['regiao'].unique())}")
        print(f" Exemplos: {', '.join(sub['municipio'].head(4).tolist())}")

    print("\n=== REGRAS DE ASSOCIAÇÃO (principais padrões) ===")
    df2["restinga_cat"] = pd.cut(df2["restinga_pct"], bins=[0, 15, 35, 100], labels=["Baixa", "Media", "Alta"])
    df2["urban_cat"] = pd.cut(df2["urbanizacao_pct"], bins=[0, 35, 60, 100], labels=["Baixa", "Media", "Alta"])

    total = len(df2)
    regras = []
    combos = [
        ("restinga_cat", "Alta", "vulnerabilidade_classe", "Baixa"),
        ("restinga_cat", "Baixa", "vulnerabilidade_classe", "Alta"),
        ("urban_cat", "Alta", "vulnerabilidade_classe", "Alta"),
        ("urban_cat", "Baixa", "vulnerabilidade_classe", "Baixa"),
        ("tem_uc", "Sim", "vulnerabilidade_classe", "Baixa"),
        ("tem_uc", "Nao", "vulnerabilidade_classe", "Alta"),
    ]

    for col_a, val_a, col_b, val_b in combos:
        mask_a = df2[col_a] == val_a
        mask_ab = mask_a & (df2[col_b] == val_b)
        suporte = mask_ab.sum() / total
        confianca = mask_ab.sum() / mask_a.sum() if mask_a.sum() > 0 else 0
        lift = confianca / (df2[col_b] == val_b).mean()
        if suporte > 0.05:
            regras.append((f"{col_a}={val_a}", f"{col_b}={val_b}", suporte, confianca, lift))

    regras.sort(key=lambda x: -x[3])
    print(f"{'Antecedente':<25} {'Consequente':<35} {'Suporte':>8} {'Confiança':>9} {'Lift':>6}")
    print("-" * 90)
    for ant, con, sup, conf, lift in regras:
        print(f"{ant:<25} → {con:<33} {sup:.2f} {conf:.2f} {lift:.2f}")


def inspect_extra_csv(path: Path) -> None:
    """Inspeção rápida do CSV extra, sem interferir no fluxo principal."""
    if not path.exists():
        print(f"\n[AVISO] CSV extra não encontrado: {path}")
        return

    print_header("CSV EXTRA OPCIONAL")
    print(f"Arquivo: {path}")
    print(f"Tamanho: {path.stat().st_size / (1024 * 1024):.1f} MB")

    try:
        preview = pd.read_csv(path, nrows=5)
        print("Colunas:", preview.columns.tolist())
        print(preview.head().to_string(index=False))
    except Exception as e:
        print(f"Não foi possível ler o CSV extra: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduz a análise do log em Python, assumindo os CSVs no mesmo diretório do script."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Nome ou caminho do CSV principal. Padrão: municipios_costeiros_vulnerabilidade.csv",
    )
    parser.add_argument(
        "--extra-csv",
        type=str,
        default=None,
        help="CSV opcional extra para inspeção.",
    )
    args = parser.parse_args()

    try:
        csv_path = resolve_file(args.csv, "municipios_costeiros_vulnerabilidade.csv")
        df = load_csv(csv_path)

        print_header("BASE CARREGADA")
        print(f"Arquivo: {csv_path}")
        print(f"Linhas: {len(df)}")
        print(f"Colunas: {len(df.columns)}")

        for col in ["vulnerabilidade_ivc", "tem_uc", "tipo_uc"]:
            if col in df.columns:
                print(f"\n{col}:")
                print(df[col].value_counts(dropna=False).to_string())

        if "tem_uc" in df.columns and "vulnerabilidade_classe" in df.columns:
            print("\nVulnerabilidade por presença de UC:")
            print(pd.crosstab(df["tem_uc"], df["vulnerabilidade_classe"]).to_string())

        if "vulnerabilidade_ivc" in df.columns and "restinga_pct" in df.columns:
            print("\nMédia de restinga por vulnerabilidade:")
            print(df.groupby("vulnerabilidade_ivc")["restinga_pct"].mean().round(1).to_string())

        exploratory_analysis(df)
        j48_like_tree(df)
        clustering_and_rules(df)

        if args.extra_csv is not None:
            extra_path = resolve_file(args.extra_csv, Path(args.extra_csv).name)
            inspect_extra_csv(extra_path)

        print("\nConcluído.")
        return 0

    except Exception as e:
        print(f"\n[ERRO] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())