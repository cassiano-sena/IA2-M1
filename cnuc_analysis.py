from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PI_CATEGORIES = {
    "estacao ecologica",
    "reserva biologica",
    "parque",
    "monumento natural",
    "refugio de vida silvestre",
}

US_CATEGORIES = {
    "area de protecao ambiental",
    "area de relevante interesse ecologico",
    "floresta",
    "reserva extrativista",
    "reserva de desenvolvimento sustentavel",
    "reserva particular do patrimonio natural",
}


def print_header(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_area_value(value) -> float:
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if not s:
        return np.nan

    s = s.replace(" ", "")
    s = re.sub(r"[^\d,.\-]", "", s)

    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") == 1:
        left, right = s.split(".")
        if len(right) == 3 and left:
            s = left + right
    elif s.count(".") > 1:
        parts = s.split(".")
        if len(parts[-1]) == 3:
            s = "".join(parts)
        else:
            s = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return float(s)
    except Exception:
        return np.nan


def standardize_group(category_manejo, group_value) -> str:
    cat = norm_text(category_manejo)
    grp = norm_text(group_value)

    if cat in PI_CATEGORIES or grp in {"pi", "protecao integral"}:
        return "PI"
    if cat in US_CATEGORIES or grp in {"us", "uso sustentavel"}:
        return "US"
    return "Desconhecido"


def detect_separator(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        raw = path.read_text(encoding="latin-1", errors="replace")
    first_line = raw.split("\n")[0]
    return "," if first_line.count(",") > first_line.count(";") else ";"


def load_csv(path: Path) -> pd.DataFrame:
    sep = detect_separator(path)
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Código UC": "codigo_uc",
        "Códogo UC": "codigo_uc",
        "C�digo UC": "codigo_uc",
        "ID_UC": "codigo_uc",
        "Nome da UC": "nome_uc",
        "NOME DA UC": "nome_uc",
        "Categoria de Manejo": "categoria_manejo",
        "Grupo": "grupo",
        "Área (ha)": "area_ha",
        "Area (ha)": "area_ha",
        "Área soma biomas": "area_ha",
        "Area soma biomas": "area_ha",
        "�rea soma biomas": "area_ha",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def extract_year_from_filename(path: Path) -> int | None:
    match = re.search(r"(20\d{2})", path.stem)
    return int(match.group(1)) if match else None


def load_folder(folder: Path) -> pd.DataFrame:
    all_dfs = []
    required = ["codigo_uc", "nome_uc", "categoria_manejo", "area_ha"]

    for file in sorted(folder.glob("*.csv")):
        year = extract_year_from_filename(file)
        if year is None:
            print(f"  [AVISO] Ano não encontrado: {file.name}", file=sys.stderr)
            continue

        try:
            df = load_csv(file)
            df = normalize_columns(df)

            if "grupo" not in df.columns:
                df["grupo"] = "Desconhecido"

            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"  [AVISO] {file.name} ignorado: faltando {missing}", file=sys.stderr)
                continue

            df["area_ha"] = df["area_ha"].apply(parse_area_value)
            df["ano"] = year

            df = df[required + ["grupo", "ano"]].copy()
            df["grupo_norm"] = df.apply(
                lambda r: standardize_group(r["categoria_manejo"], r["grupo"]),
                axis=1,
            )

            df = df.dropna(subset=["codigo_uc", "nome_uc", "categoria_manejo", "area_ha"])
            df["codigo_uc"] = df["codigo_uc"].astype(str).str.strip()
            df["nome_uc"] = df["nome_uc"].astype(str).str.strip()
            df["categoria_manejo"] = df["categoria_manejo"].astype(str).str.strip()

            all_dfs.append(df.reset_index(drop=True))
            print(f"  [OK] {file.name} ({year}): {len(df)} registros")

        except Exception as e:
            print(f"  [ERRO] {file.name}: {e}", file=sys.stderr)

    if not all_dfs:
        raise ValueError("Nenhum CSV válido encontrado")

    return pd.concat(all_dfs, ignore_index=True).reset_index(drop=True)


def build_variation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_year = int(df["ano"].min())

    for codigo_uc, g in df.groupby("codigo_uc", sort=False):
        g = g.sort_values("ano").dropna(subset=["area_ha"])
        g_year = g.groupby("ano", as_index=False)["area_ha"].median().sort_values("ano")

        if g_year["ano"].nunique() < 2:
            continue

        years = g_year["ano"].astype(int).to_numpy()
        areas = g_year["area_ha"].astype(float).to_numpy()

        first_seen = int(years.min())
        last_seen = int(years.max())
        observed_years = int(len(years))
        span_years = int(last_seen - first_seen + 1)
        coverage_ratio = observed_years / span_years if span_years > 0 else np.nan

        start_n = min(2, len(areas))
        end_n = min(2, len(areas))
        area_inicial = float(np.median(areas[:start_n]))
        area_final = float(np.median(areas[-end_n:]))
        delta_area = area_final - area_inicial
        delta_pct = np.nan if area_inicial <= 0 else delta_area / area_inicial

        slope_ha_ano = np.nan
        slope_pct_ano = np.nan
        if len(areas) >= 2 and np.isfinite(areas).sum() >= 2:
            slope_ha_ano = stats.linregress(years, areas).slope
            media_area = float(np.mean(areas))
            if media_area > 0:
                slope_pct_ano = slope_ha_ano / media_area

        yoy = np.diff(areas) / np.where(areas[:-1] > 0, areas[:-1], np.nan)
        max_abs_yoy_pct = np.nanmax(np.abs(yoy)) if len(yoy) else np.nan
        median_abs_yoy_pct = np.nanmedian(np.abs(yoy)) if len(yoy) else np.nan

        base = g.iloc[0]
        rows.append({
            "codigo_uc": codigo_uc,
            "nome_uc": base["nome_uc"],
            "categoria_manejo": base["categoria_manejo"],
            "grupo": base["grupo_norm"],
            "ano_inicial": first_seen,
            "ano_final": last_seen,
            "area_inicial": area_inicial,
            "area_final": area_final,
            "delta_area": delta_area,
            "delta_pct": delta_pct,
            "slope_ha_ano": slope_ha_ano,
            "slope_pct_ano": slope_pct_ano,
            "n_anos": observed_years,
            "span_anos": span_years,
            "coverage_ratio": coverage_ratio,
            "max_abs_yoy_pct": max_abs_yoy_pct,
            "median_abs_yoy_pct": median_abs_yoy_pct,
            "coorte": "nova" if first_seen > base_year else "existente",
            "serie_tipo": "curta" if observed_years < 4 else "longa",
        })

    df_var = pd.DataFrame(rows).dropna(subset=["delta_pct"]).copy()

    df_var["qc_area_suspeita"] = (df_var["area_inicial"] < 5) | (df_var["area_final"] < 5)
    df_var["qc_salto_excessivo"] = df_var["delta_pct"].abs() > 5
    df_var["qc_yoy_excessivo"] = df_var["max_abs_yoy_pct"].fillna(0) > 5
    df_var["qc_cobertura_baixa"] = df_var["coverage_ratio"].fillna(0) < 0.5
    df_var["qc_pass"] = ~(
        df_var["qc_area_suspeita"]
        | df_var["qc_salto_excessivo"]
        | df_var["qc_yoy_excessivo"]
    )

    return df_var


def cohort_analysis(df: pd.DataFrame) -> None:
    print_header("ANÁLISE POR COORTE")

    df_var = build_variation_table(df)
    clean = df_var[df_var["qc_pass"]].copy()

    print("Distribuição por coorte:")
    print(clean["coorte"].value_counts().to_string())

    print("\nDistribuição por tipo de série:")
    print(clean["serie_tipo"].value_counts().to_string())

    for col in ["delta_pct", "slope_pct_ano", "median_abs_yoy_pct", "coverage_ratio"]:
        nova = clean.loc[clean["coorte"] == "nova", col].dropna()
        antiga = clean.loc[clean["coorte"] == "existente", col].dropna()

        if len(nova) >= 5 and len(antiga) >= 5:
            stat, p = stats.mannwhitneyu(nova, antiga, alternative="two-sided")
            print(f"\n{col}: nova={nova.median():.4f} | existente={antiga.median():.4f} | p={p:.4f}")


def exploratory_analysis(df: pd.DataFrame) -> None:
    print_header("ANÁLISE EXPLORATÓRIA")

    df_var = build_variation_table(df)
    clean = df_var[df_var["qc_pass"]].copy()

    print(f"UCs com série temporal (>= 2 anos): {len(df_var)}")
    print(f"UCs aprovadas no controle de qualidade: {len(clean)}")
    print(f"Período coberto: {int(df['ano'].min())} – {int(df['ano'].max())}")

    print("\n--- VARIAÇÃO DE ÁREA (limpa) ---")
    print(clean["delta_pct"].describe())

    print("\n--- TENDÊNCIA ---")
    clean["tendencia"] = pd.cut(
        clean["delta_pct"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=["reduziu", "estavel", "aumentou"],
    )
    print(clean["tendencia"].value_counts().to_string())

    print("\n--- VARIAÇÃO MÉDIA POR CATEGORIA DE MANEJO ---")
    print(clean.groupby("categoria_manejo")["delta_pct"].median().sort_values().to_string())

    print("\n--- VARIAÇÃO MÉDIA POR GRUPO ---")
    print(clean.groupby("grupo")["delta_pct"].median().to_string())

    print("\n--- VARIAÇÃO MÉDIA POR COORTE ---")
    print(clean.groupby("coorte")["delta_pct"].median().to_string())

    print("\n--- VARIAÇÃO MÉDIA POR TIPO DE SÉRIE ---")
    print(clean.groupby("serie_tipo")["delta_pct"].median().to_string())

    print("\n--- TOP 10 REDUÇÕES APÓS QC ---")
    print(
        clean.nsmallest(10, "delta_pct")[
            ["nome_uc", "categoria_manejo", "grupo", "coorte", "serie_tipo", "area_inicial", "area_final", "delta_pct"]
        ].to_string(index=False)
    )

    print("\n--- TOP 10 EXPANSÕES APÓS QC ---")
    print(
        clean.nlargest(10, "delta_pct")[
            ["nome_uc", "categoria_manejo", "grupo", "coorte", "serie_tipo", "area_inicial", "area_final", "delta_pct"]
        ].to_string(index=False)
    )


def j48_like_tree(df: pd.DataFrame) -> None:
    print_header("ÁRVORE DE DECISÃO (J48-like)")

    df_var = build_variation_table(df)
    df_var = df_var[df_var["qc_pass"]].copy()

    df_var["classe"] = pd.cut(
        df_var["delta_pct"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=["reduziu", "estavel", "aumentou"],
    )
    df_var = df_var.dropna(subset=["classe"]).copy()

    if df_var["classe"].nunique() < 2:
        print("Dados insuficientes para treinar a árvore.")
        return

    X = pd.get_dummies(
        df_var[["categoria_manejo", "grupo", "coorte", "serie_tipo"]],
        drop_first=False,
    )
    y = df_var["classe"]

    clf = DecisionTreeClassifier(
        criterion="entropy",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
    )

    n_splits = min(5, y.value_counts().min())
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        print(f"F1 macro média: {scores.mean():.3f} (±{scores.std():.3f})")
        print("\nMatriz de confusão:")
        print(confusion_matrix(y, y_pred))
        print("\nRelatório de classificação:")
        print(classification_report(y, y_pred, zero_division=0))
    else:
        print("Não há amostras suficientes por classe para validação cruzada.")

    clf.fit(X, y)

    print("\n--- ESTRUTURA DA ÁRVORE ---")
    print(export_text(clf, feature_names=list(X.columns)))

    print("\n--- IMPORTÂNCIA DOS ATRIBUTOS ---")
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"{feat:<35}: {imp:.3f}")


def clustering(df: pd.DataFrame) -> None:
    print_header("CLUSTERIZAÇÃO (K-Means)")

    df_var = build_variation_table(df)
    df_var = df_var[df_var["qc_pass"]].copy()

    features = pd.DataFrame({
        "delta_pct": df_var["delta_pct"].clip(-1, 1),
        "slope_pct_ano": df_var["slope_pct_ano"].fillna(0).clip(-1, 1),
        "coverage_ratio": df_var["coverage_ratio"].fillna(0).clip(0, 1),
        "log_area_inicial": np.log1p(df_var["area_inicial"]),
        "log_area_final": np.log1p(df_var["area_final"]),
    })

    X_scaled = RobustScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    df_var["cluster"] = kmeans.fit_predict(X_scaled)

    for c in sorted(df_var["cluster"].unique()):
        sub = df_var[df_var["cluster"] == c]
        print(f"\nCluster {c} ({len(sub)} UCs)")
        print(f"  delta_pct mediano   : {sub['delta_pct'].median():.4f}")
        print(f"  slope_pct_ano med   : {sub['slope_pct_ano'].median():.4f}")
        print(f"  cobertura mediana   : {sub['coverage_ratio'].median():.2f}")
        print(f"  área inicial med    : {sub['area_inicial'].median():,.1f} ha")
        print(f"  área final med      : {sub['area_final'].median():,.1f} ha")
        print(f"  coortes             : {sub['coorte'].value_counts().to_dict()}")
        print(f"  grupos              : {sub['grupo'].value_counts().to_dict()}")
        print(f"  categorias top      : {sub['categoria_manejo'].value_counts().head(3).to_dict()}")


def association_rules(df: pd.DataFrame) -> None:
    print_header("REGRAS DE ASSOCIAÇÃO (simplificadas)")

    df_var = build_variation_table(df)
    df_var = df_var[df_var["qc_pass"]].copy()

    df_var["tendencia"] = pd.cut(
        df_var["delta_pct"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=["reduziu", "estavel", "aumentou"],
    )

    total = len(df_var)
    combos = [
        ("grupo", "PI", "tendencia", "reduziu"),
        ("grupo", "US", "tendencia", "reduziu"),
        ("coorte", "nova", "tendencia", "aumentou"),
        ("coorte", "existente", "tendencia", "estavel"),
        ("serie_tipo", "curta", "tendencia", "reduziu"),
        ("serie_tipo", "longa", "tendencia", "estavel"),
    ]

    print(f"{'Antecedente':<30} {'Consequente':<20} {'Suporte':>8} {'Confiança':>9} {'Lift':>6}")
    print("-" * 82)

    for col_a, val_a, col_b, val_b in combos:
        mask_a = df_var[col_a].astype(str) == val_a
        mask_ab = mask_a & (df_var[col_b].astype(str) == val_b)

        suporte = mask_ab.sum() / total if total else 0
        confianca = mask_ab.sum() / mask_a.sum() if mask_a.sum() > 0 else 0
        base_rate = (df_var[col_b].astype(str) == val_b).mean()
        lift = confianca / base_rate if base_rate > 0 else 0

        if suporte > 0.02:
            print(f"{col_a}={val_a:<24} {col_b}={val_b:<14} {suporte:>8.2f} {confianca:>9.2f} {lift:>6.2f}")


def export_results(df: pd.DataFrame, out_dir: str = "results") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_var = build_variation_table(df)
    clean = df_var[df_var["qc_pass"]].copy()
    rejected = df_var[~df_var["qc_pass"]].copy()

    df.to_csv(out_path / "base_unificada.csv", index=False, sep=";", encoding="utf-8-sig")
    df_var.to_csv(out_path / "variacao_completa.csv", index=False, sep=";", encoding="utf-8-sig")
    clean.to_csv(out_path / "variacao_limpa.csv", index=False, sep=";", encoding="utf-8-sig")
    rejected.to_csv(out_path / "variacao_rejeitada_qc.csv", index=False, sep=";", encoding="utf-8-sig")

    clean["tendencia"] = pd.cut(
        clean["delta_pct"],
        bins=[-np.inf, -0.01, 0.01, np.inf],
        labels=["reduziu", "estavel", "aumentou"],
    )

    resumo_categoria = (
        clean.groupby("categoria_manejo")
        .agg(
            n=("codigo_uc", "count"),
            delta_pct_mediana=("delta_pct", "median"),
            delta_pct_media=("delta_pct", "mean"),
            area_inicial_mediana=("area_inicial", "median"),
            area_final_mediana=("area_final", "median"),
        )
        .reset_index()
        .sort_values("delta_pct_mediana")
    )
    resumo_categoria.to_csv(out_path / "resumo_categoria_manejo.csv", index=False, sep=";", encoding="utf-8-sig")

    resumo_grupo = (
        clean.groupby("grupo")
        .agg(
            n=("codigo_uc", "count"),
            delta_pct_mediana=("delta_pct", "median"),
            delta_pct_media=("delta_pct", "mean"),
            area_inicial_mediana=("area_inicial", "median"),
            area_final_mediana=("area_final", "median"),
        )
        .reset_index()
        .sort_values("delta_pct_mediana")
    )
    resumo_grupo.to_csv(out_path / "resumo_grupo.csv", index=False, sep=";", encoding="utf-8-sig")

    resumo_coorte = (
        clean.groupby("coorte")
        .agg(
            n=("codigo_uc", "count"),
            delta_pct_mediana=("delta_pct", "median"),
            slope_pct_ano_mediana=("slope_pct_ano", "median"),
            cobertura_mediana=("coverage_ratio", "median"),
        )
        .reset_index()
    )
    resumo_coorte.to_csv(out_path / "resumo_coorte.csv", index=False, sep=";", encoding="utf-8-sig")

    resumo_serie = (
        clean.groupby("serie_tipo")
        .agg(
            n=("codigo_uc", "count"),
            delta_pct_mediana=("delta_pct", "median"),
            slope_pct_ano_mediana=("slope_pct_ano", "median"),
            cobertura_mediana=("coverage_ratio", "median"),
        )
        .reset_index()
    )
    resumo_serie.to_csv(out_path / "resumo_serie_tipo.csv", index=False, sep=";", encoding="utf-8-sig")

    clean.nsmallest(25, "delta_pct").to_csv(out_path / "top_25_reducoes.csv", index=False, sep=";", encoding="utf-8-sig")
    clean.nlargest(25, "delta_pct").to_csv(out_path / "top_25_expansoes.csv", index=False, sep=";", encoding="utf-8-sig")

    print(f"\nArquivos exportados para: {out_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Análise de variação de área das UCs do CNUC (2018–2025).")
    parser.add_argument("--folder", type=str, default=".", help="Pasta com os CSVs do CNUC.")
    args = parser.parse_args()

    try:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"[ERRO] Pasta não encontrada: {folder}", file=sys.stderr)
            return 1

        print(f"Carregando CSVs de: {folder.resolve()}")
        df = load_folder(folder)
        export_results(df)

        print_header("BASE UNIFICADA")
        print(f"Total de linhas      : {len(df)}")
        print(f"UCs únicas           : {df['codigo_uc'].nunique()}")
        print(f"Anos disponíveis     : {[int(x) for x in sorted(df['ano'].unique())]}")
        print(f"Categorias de manejo : {df['categoria_manejo'].nunique()}")

        exploratory_analysis(df)
        cohort_analysis(df)
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