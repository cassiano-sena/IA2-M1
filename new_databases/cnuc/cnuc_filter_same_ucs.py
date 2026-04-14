from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


"""Filtra CSVs do CNUC mantendo apenas as UCs presentes em um CSV-base.

Uso típico:
    python cnuc_filter_same_ucs.py \
        --base /mnt/data/cnuc_2025_ucs_municipios_costeiros.csv \
        --input /caminho/para/outro_cnuc_2024.csv \
        --output-dir /caminho/saida

Também aceita uma pasta com vários CSVs:
    python cnuc_filter_same_ucs.py --base base.csv --input-dir ./cnuc_csvs --output-dir ./saida

O script:
- lê o CSV-base e extrai os códigos das UCs;
- percorre um ou mais CSVs do CNUC de mesma autoria;
- mantém apenas as linhas cujas UCs existem no CSV-base;
- não inclui linhas ausentes;
- preserva o padrão do arquivo de entrada (colunas, ordem, delimitador e codificação de saída em UTF-8 com BOM).
"""

CODE_COLUMNS_CANDIDATES = [
    "Código UC",
    "Codigo UC",
    "ID_UC",
    "ID UC",
    "CODIGO UC",
    "codigo uc",
    "código uc",
]


def detect_delimiter(path: Path, sample_size: int = 8192) -> str:
    """Detecta delimitador CSV com fallback seguro."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(sample_size)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # Fallback mais comum em CSVs do CNUC/planilhas exportadas.
        if sample.count(";") > sample.count(","):
            return ";"
        return ","


def find_code_column(columns: Iterable[str]) -> Optional[str]:
    """Encontra a coluna de código da UC em nomes possivelmente variados."""
    cols = list(columns)
    normalized = {c.strip().lower(): c for c in cols}

    for candidate in CODE_COLUMNS_CANDIDATES:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    # Tenta uma busca mais flexível removendo acentos e espaços.
    def simplify(text: str) -> str:
        import unicodedata

        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return " ".join(text.lower().replace("_", " ").split())

    simplified = {simplify(c): c for c in cols}
    for candidate in CODE_COLUMNS_CANDIDATES:
        key = simplify(candidate)
        if key in simplified:
            return simplified[key]

    return None


def load_csv(path: Path) -> pd.DataFrame:
    delimiter = detect_delimiter(path)
    try:
        return pd.read_csv(path, sep=delimiter, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=delimiter, dtype=str, encoding="latin1")


def normalize_code_series(series: pd.Series) -> pd.Series:
    """Normaliza códigos para comparação segura.

    Remove espaços e mantém texto como string.
    """
    return series.astype(str).str.strip()


def read_base_codes(base_path: Path) -> set[str]:
    df_base = load_csv(base_path)
    code_col = find_code_column(df_base.columns)
    if code_col is None:
        raise ValueError(
            f"Não encontrei coluna de código da UC no arquivo-base: {base_path.name}. "
            f"Colunas disponíveis: {list(df_base.columns)}"
        )

    codes = normalize_code_series(df_base[code_col])
    codes = codes[codes.notna()]
    codes = codes[codes != "nan"]
    return set(codes.tolist())


def filter_csv_by_codes(input_path: Path, base_codes: set[str], output_dir: Path) -> Path:
    delimiter = detect_delimiter(input_path)
    df = load_csv(input_path)

    code_col = find_code_column(df.columns)
    if code_col is None:
        raise ValueError(
            f"Não encontrei coluna de código da UC em {input_path.name}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    codes = normalize_code_series(df[code_col])
    mask = codes.isin(base_codes)
    filtered = df.loc[mask].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{input_path.stem}_filtrado.csv"
    out_path = output_dir / out_name

    # Preserva o padrão estrutural do arquivo de entrada: mesmas colunas e ordem.
    filtered.to_csv(
        out_path,
        index=False,
        sep=delimiter,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    return out_path


def gather_input_files(input_paths: list[Path], input_dir: Optional[Path], base_path: Path) -> list[Path]:
    files: list[Path] = []

    for p in input_paths:
        if p.is_dir():
            files.extend(sorted(x for x in p.glob("*.csv") if x.resolve() != base_path.resolve()))
        elif p.is_file():
            if p.resolve() != base_path.resolve():
                files.append(p)
        else:
            raise FileNotFoundError(f"Arquivo ou pasta não encontrado: {p}")

    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")
        files.extend(sorted(x for x in input_dir.glob("*.csv") if x.resolve() != base_path.resolve()))

    # Remove duplicados preservando ordem.
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            unique_files.append(f)

    return unique_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filtra CSVs do CNUC mantendo apenas as UCs presentes em um CSV-base."
    )
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="CSV-base com a lista de UCs que devem ser mantidas.",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        type=Path,
        default=[],
        help="Um ou mais CSVs de entrada do CNUC para filtrar.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Pasta contendo CSVs do CNUC para filtrar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./saida_filtrada"),
        help="Pasta de saída para os CSVs filtrados.",
    )

    args = parser.parse_args()

    if not args.base.exists():
        print(f"[ERRO] CSV-base não encontrado: {args.base}", file=sys.stderr)
        return 1

    try:
        base_codes = read_base_codes(args.base)
        if not base_codes:
            raise ValueError("O CSV-base não possui códigos válidos de UC.")

        input_files = gather_input_files(args.input, args.input_dir, args.base)
        if not input_files:
            print("[AVISO] Nenhum CSV de entrada informado para filtrar.")
            return 0

        print(f"Base carregada: {args.base.name}")
        print(f"Quantidade de UCs-base: {len(base_codes)}")
        print(f"Arquivos de entrada: {len(input_files)}")

        for input_path in input_files:
            try:
                out_path = filter_csv_by_codes(input_path, base_codes, args.output_dir)

                # Relatório rápido de contagem.
                original_df = load_csv(input_path)
                code_col = find_code_column(original_df.columns)
                original_count = len(original_df)
                kept_count = len(load_csv(out_path))

                print(
                    f"[OK] {input_path.name} -> {out_path.name} | "
                    f"mantidas {kept_count} de {original_count} linhas | coluna usada: {code_col}"
                )
            except Exception as e:
                print(f"[ERRO] Falha ao processar {input_path.name}: {e}", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
