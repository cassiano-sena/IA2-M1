from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

"""
Análise a partir de CSVs do CNUC (Cadastro Nacional de Unidades de Conservação).

Fluxo principal:
    1) Lê todos os CSVs de uma pasta (um por ano, 2018–2025)
    2) Normaliza colunas inconsistentes entre anos
    3) Executa análises exploratórias de variação de área
    4) Treina uma árvore de decisão estilo J48/WEKA
    5) Executa K-Means e regras de associação simplificadas

Uso:
    ./cnuc_analysis.py --folder /pasta/com/csvs/
    ./cnuc_analysis.py --folder .
"""

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def detect_separator(path: Path) -> str:
    """Detecta o separador real do arquivo pelo conteúdo da primeira linha."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        raw = path.read_text(encoding="latin-1", errors="replace")
    first_line = raw.split("\n")[0]
    return "," if first_line.count(",") > first_line.count(";") else ";"


def load_csv(path: Path) -> pd.DataFrame:
    """Lê CSV detectando separador e encoding automaticamente."""
    sep = detect_separator(path)
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    # último recurso: leitura permissiva
    return pd.read_csv(path, sep=sep, encoding="utf-8", errors="replace", low_memory=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia colunas para nomes canônicos, cobrindo todas as variações
    encontradas nos CSVs do CNUC de 2018 a 2025.
    """
    rename_map = {
        # --- código da UC ---
        "Código UC":   "codigo_uc",
        "C\u00f3digo UC": "codigo_uc",   # UTF-8 correto
        "C\ufffdigo UC": "codigo_uc",    # replacement character
        "ID_UC":       "codigo_uc",

        # --- nome da UC ---
        "Nome da UC":  "nome_uc",
        "NOME DA UC":  "nome_uc",
        "Nome da UC ": "nome_uc",        # trailing space

        # --- categoria de manejo ---
        "Categoria de Manejo": "categoria_manejo",

        # --- grupo (PI / US) ---
        "Grupo": "grupo",

        # --- área ---
        # 2018 usa "Área (ha)", demais anos usam "Área soma biomas"
        "Área (ha)":       "area_ha",
        "\u00c1rea (ha)":  "area_ha",    # Á em UTF-8
        "\ufffdrea (ha)":  "area_ha",    # encoding corrompido
        "Area (ha)":       "area_ha",    # sem acento
        "Área soma biomas":   "area_ha",
        "\u00c1rea soma biomas": "area_ha",
        "\ufffdrea soma biomas": "area_ha",
        "Area soma biomas":    "area_ha",
    }

    # aplica apenas as colunas que existem no df atual
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


def extract_year_from_filename(path: Path) -> int | None:
    """Extrai o ano (20XX) do nome do arquivo via regex."""
    match = re.search(r"(20\d{2})", path.stem)
    if match:
        return int(match.group(1))
    print(f"  [AVISO] Ano não encontrado no nome do arquivo: {path.name}",
          file=sys.stderr)
    return None


def load_folder(folder: Path) -> pd.DataFrame:
    """
    Carrega todos os CSVs válidos de uma pasta, normaliza colunas e
    retorna um DataFrame unificado com coluna 'ano'.
    """
    all_dfs: list[pd.DataFrame] = []
    required = ["codigo_uc", "nome_uc", "categoria_manejo", "area_ha"]
    counter = 0

    for file in sorted(folder.glob("*.csv")):
        year = extract_year_from_filename(file)
        if year is None:
            continue

        try:
            df = load_csv(file)
            df = normalize_columns(df)

            # 'grupo' é opcional — ausente no CSV de 2018
            if "grupo" not in df.columns:
                df["grupo"] = "Desconhecido"

            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"  [AVISO] {file.name} ({year}): colunas ausentes após "
                      f"normalização: {missing}. Arquivo ignorado.", file=sys.stderr)
                continue

            # converte área para numérico (pode vir com vírgula decimal)
            df["area_ha"] = (
                df["area_ha"]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(r"\s", "", regex=True)
            )
            df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce")

            df = df[required + ["grupo", "ano"] if "ano" in df.columns
                    else required + ["grupo"]].copy()
            df = df.reset_index(drop=True)
            df["ano"] = year
            df = df.dropna(subset=required)

            df = df.drop_duplicates()
            df = df.reset_index(drop=True)
            df = df.astype({"codigo_uc": str, "nome_uc": str, "categoria_manejo": str, "area_ha": float, "grupo": str, "ano": int})
            df.index = range(counter, counter + len(df))
            counter += len(df)
            df = df.reset_index(drop=True)
            all_dfs.append(df)
            print(f"  [OK] {file.name} ({year}): {len(df)} registros")

        except Exception as exc:
            print(f"  [ERRO] {file.name}: {exc}", file=sys.stderr)
            continue

    if not all_dfs:
        raise ValueError(
            "Nenhum CSV válido encontrado na pasta. "
            "Verifique se os arquivos contêm os anos (20XX) no nome."
        )

    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def build_variation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega por UC: área inicial, área final, delta absoluto e delta percentual.
    Inclui apenas UCs com pelo menos 2 anos distintos de dados.
    """
    df_sorted = df.sort_values(["codigo_uc", "ano"])

    df_var = df_sorted.groupby("codigo_uc").agg(
        nome_uc=("nome_uc", "first"),
        categoria_manejo=("categoria_manejo", "first"),
        grupo=("grupo", "first"),
        ano_inicial=("ano", "min"),
        ano_final=("ano", "max"),
        area_inicial=("area_ha", "first"),
        area_final=("area_ha", "last"),
        n_anos=("ano", "nunique"),
    ).reset_index()

    # somente UCs com série temporal real (>= 2 anos distintos)
    df_var = df_var[df_var["n_anos"] >= 2].copy()

    df_var["delta_area"] = df_var["area_final"] - df_var["area_inicial"]

    # proteção contra divisão por zero
    df_var["delta_pct"] = np.where(
        df_var["area_inicial"] > 0,
        df_var["delta_area"] / df_var["area_inicial"],
        np.nan,
    )
    df_var = df_var.dropna(subset=["delta_pct"])

    return df_var


# ---------------------------------------------------------------------------
# Análise exploratória
# ---------------------------------------------------------------------------

def exploratory_analysis(df: pd.DataFrame) -> None:
    print_header("ANÁLISE EXPLORATÓRIA")

    df_var = build_variation_table(df)

    print(f"\nUCs com série temporal (>= 2 anos): {len(df_var)}")
    print(f"Período coberto: {df['ano'].min()} – {df['ano'].max()}")

    print("\n--- VARIAÇÃO DE ÁREA (delta_pct) ---")
    print(df_var["delta_pct"].describe())

    print("\n--- CLASSIFICAÇÃO POR TENDÊNCIA ---")
    df_var["tendencia"] = pd.cut(
        df_var["delta_pct"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=["reduziu", "estavel", "aumentou"],
    )
    print(df_var["tendencia"].value_counts())

    print("\n--- VARIAÇÃO MÉDIA POR CATEGORIA DE MANEJO ---")
    print(
        df_var.groupby("categoria_manejo")["delta_pct"]
        .mean()
        .sort_values()
        .to_string()
    )

    print("\n--- VARIAÇÃO MÉDIA POR GRUPO (PI / US) ---")
    print(df_var.groupby("grupo")["delta_pct"].mean().to_string())

    print("\n--- TOP 10 UCs COM MAIOR REDUÇÃO ---")
    print(
        df_var.nsmallest(10, "delta_pct")[
            ["nome_uc", "categoria_manejo", "grupo",
             "area_inicial", "area_final", "delta_pct"]
        ].to_string(index=False)
    )

    print("\n--- TOP 10 UCs COM MAIOR EXPANSÃO ---")
    print(
        df_var.nlargest(10, "delta_pct")[
            ["nome_uc", "categoria_manejo", "grupo",
             "area_inicial", "area_final", "delta_pct"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Árvore de decisão (J48-like)
# ---------------------------------------------------------------------------

def j48_like_tree(df: pd.DataFrame) -> None:
    print_header("ÁRVORE DE DECISÃO (J48-like)")

    df_var = build_variation_table(df)

    df_var["classe"] = pd.cut(
        df_var["delta_pct"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=["reduziu", "estavel", "aumentou"],
    )

    le_cat = LabelEncoder()
    le_grp = LabelEncoder()

    X = pd.DataFrame({
        "categoria_manejo": le_cat.fit_transform(df_var["categoria_manejo"]),
        "grupo": le_grp.fit_transform(df_var["grupo"]),
    })

    y = df_var["classe"]

    if y.nunique() < 2:
        print("Dados insuficientes para treinar a árvore (menos de 2 classes).")
        return

    clf = DecisionTreeClassifier(
        criterion="entropy",
        min_samples_leaf=2,
        ccp_alpha=0.01,
        random_state=SEED,
    )
    clf.fit(X, y)

    print("\n--- ESTRUTURA DA ÁRVORE ---")
    print(export_text(clf, feature_names=list(X.columns)))

    print("--- DISTRIBUIÇÃO DAS CLASSES ---")
    print(y.value_counts().to_string())

    print("\n--- IMPORTÂNCIA DOS ATRIBUTOS ---")
    for feat, imp in zip(X.columns, clf.feature_importances_):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25}: {imp:.3f}  {bar}")


# ---------------------------------------------------------------------------
# Clusterização
# ---------------------------------------------------------------------------

def clustering(df: pd.DataFrame) -> None:
    print_header("CLUSTERIZAÇÃO (K-Means, k=3)")

    df_var = build_variation_table(df)

    # usa 3 dimensões para clusters mais informativos
    features = ["delta_pct", "area_inicial", "area_final"]
    X = df_var[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    df_var["cluster"] = kmeans.fit_predict(X_scaled)

    for c in sorted(df_var["cluster"].unique()):
        sub = df_var[df_var["cluster"] == c]
        print(f"\nCluster {c}  ({len(sub)} UCs)")
        print(f"  delta_pct médio : {sub['delta_pct'].mean():.4f}")
        print(f"  área inicial méd: {sub['area_inicial'].mean():,.1f} ha")
        print(f"  área final méd  : {sub['area_final'].mean():,.1f} ha")
        print(f"  grupos          : {sub['grupo'].value_counts().to_dict()}")
        print(f"  top categorias  : "
              f"{sub['categoria_manejo'].value_counts().head(3).to_dict()}")


# ---------------------------------------------------------------------------
# Regras de associação simplificadas
# ---------------------------------------------------------------------------

def association_rules(df: pd.DataFrame) -> None:
    print_header("REGRAS DE ASSOCIAÇÃO (simplificadas)")

    df_var = build_variation_table(df)

    df_var["tendencia"] = pd.cut(
        df_var["delta_pct"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=["reduziu", "estavel", "aumentou"],
    )

    total = len(df_var)
    combos = [
        ("grupo", "PI",  "tendencia", "reduziu"),
        ("grupo", "US",  "tendencia", "reduziu"),
        ("grupo", "PI",  "tendencia", "estavel"),
        ("grupo", "US",  "tendencia", "estavel"),
        ("grupo", "Desconhecido", "tendencia", "estavel"),
    ]

    header = f"{'Antecedente':<30} {'Consequente':<20} {'Suporte':>8} {'Confiança':>9} {'Lift':>6}"
    print(header)
    print("-" * len(header))

    for col_a, val_a, col_b, val_b in combos:
        mask_a = df_var[col_a].astype(str) == val_a
        mask_ab = mask_a & (df_var[col_b].astype(str) == val_b)

        suporte = mask_ab.sum() / total
        confianca = mask_ab.sum() / mask_a.sum() if mask_a.sum() > 0 else 0
        base_rate = (df_var[col_b].astype(str) == val_b).mean()
        lift = confianca / base_rate if base_rate > 0 else 0

        if suporte > 0.02:
            print(
                f"{col_a}={val_a:<24} {col_b}={val_b:<14} "
                f"{suporte:>8.2f} {confianca:>9.2f} {lift:>6.2f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Análise de variação de área das UCs do CNUC (2018–2025)."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=".",
        help="Pasta contendo os CSVs do CNUC, um por ano (padrão: diretório atual).",
    )
    args = parser.parse_args()

    try:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"[ERRO] Pasta não encontrada: {folder}", file=sys.stderr)
            return 1

        print(f"Carregando CSVs de: {folder.resolve()}")
        df = load_folder(folder)

        print_header("BASE UNIFICADA")
        print(f"Total de linhas      : {len(df)}")
        print(f"UCs únicas           : {df['codigo_uc'].nunique()}")
        print(f"Anos disponíveis     : {sorted(df['ano'].unique())}")
        print(f"Categorias de manejo : {df['categoria_manejo'].nunique()}")

        exploratory_analysis(df)
        j48_like_tree(df)
        clustering(df)
        association_rules(df)

        print("\nConcluído.")
        return 0

    except Exception as e:
        print(f"\n[ERRO] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())