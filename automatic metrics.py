import pandas as pd
from easse.easse import bleu,sari
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
import torch
import argparse
import os
from typing import Dict, Iterable, Optional

def calculate_sari(reference: str, prediction: str, original: str) -> float:
    """Compute corpus-level SARI for a single example.

    Args:
        reference: Reference simplification.
        prediction: Model prediction.
        original: Original (complex) sentence.

    Returns:
        SARI score.
    """
    refs_sents = [[str(reference)]]
    orig_sents = [str(original)]
    sys_sents = [str(prediction)]
    return sari.corpus_sari(refs_sents=refs_sents, sys_sents=sys_sents, orig_sents=orig_sents)

def calculate_bleu(reference: str, prediction: str) -> float:
    """Compute corpus-level BLEU for a single example.

    Args:
        reference: Reference text.
        prediction: Predicted text.

    Returns:
        BLEU score.
    """
    sys_sents = [str(prediction)]
    refs_sents = [[str(reference)]]
    return bleu.corpus_bleu(refs_sents=refs_sents, sys_sents=sys_sents)

def calculate_camembert_similarity(
    model: SentenceTransformer,
    reference: str,
    prediction: str,
) -> float:
    """Compute CamemBERT cosine similarity.

    Args:
        model: A loaded `SentenceTransformer` model.
        reference: Reference text.
        prediction: Predicted text.

    Returns:
        Mean cosine similarity.
    """
    sentence_embedding = model.encode(str(prediction), convert_to_tensor=True)
    reference_embedding = model.encode(str(reference), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(sentence_embedding, reference_embedding)
    return float(torch.mean(cosine_scores).item())

DEFAULT_DATASETS = ["Clear", "WikiLarge FR", "asset", "MultiCochrane", "WikiAuto"]


def infer_language_and_columns(dataset_name: str) -> tuple[str, str, str]:
    """Infer language and column names for a dataset.

    Args:
        dataset_name: Dataset basename.

    Returns:
        (language, reference_col, complex_col)
    """
    if dataset_name in ["Clear", "WikiLarge FR"]:
        return ("english", "English Simplified", "English Translated")
    return ("french", "French Simplified", "French Translated")


def _pick_prediction_column(df: pd.DataFrame) -> str:
    cols = [col for col in df.columns if "response" in str(col).lower()]
    if not cols:
        raise ValueError("No prediction column found (expected a column containing 'response').")
    return str(cols[0])


def run(
    datasets: Iterable[str],
    input_dir: str = "llm output",
    output_dir: str = "automatic metrics results",
    camembert_model_name: str = "dangvantuan/sentence-camembert-base",
) -> None:
    """Compute automatic metrics per sheet and write an output workbook.

    Args:
        datasets: Dataset basenames (without extension).
        input_dir: Directory containing `"<dataset> output.xlsx"` inputs.
        output_dir: Directory to write results to.
        camembert_model_name: SentenceTransformer model name for French similarity.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lazy-load heavy models only if needed.
    bert_scorer: Optional[BERTScorer] = None
    camembert_model: Optional[SentenceTransformer] = None

    for dataset in datasets:
        in_path = os.path.join(input_dir, f"{dataset} output.xlsx")
        xls = pd.ExcelFile(in_path)
        language, reference_col, complex_col = infer_language_and_columns(dataset)

        out_sheets: Dict[str, pd.DataFrame] = {}
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            prediction_col = _pick_prediction_column(df)

            df["bleu_score"] = df.apply(lambda row: calculate_bleu(row[reference_col], row[prediction_col]), axis=1)
            df["sari_score"] = df.apply(
                lambda row: calculate_sari(row[reference_col], row[prediction_col], row[complex_col]),
                axis=1,
            )

            if language == "english":
                if bert_scorer is None:
                    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                _, _, f1 = bert_scorer.score(
                    cands=df[prediction_col].astype(str).tolist(),
                    refs=df[reference_col].astype(str).tolist(),
                )
                df["bert_score"] = f1
            else:
                if camembert_model is None:
                    camembert_model = SentenceTransformer(camembert_model_name)
                df["camembert_score"] = df.apply(
                    lambda row: calculate_camembert_similarity(camembert_model, row[reference_col], row[prediction_col]),
                    axis=1,
                )

            out_sheets[sheet_name] = df

        out_path = os.path.join(output_dir, f"{dataset} with automatic metrics.xlsx")
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            for sheet_name, sheet_df in out_sheets.items():
                pd.DataFrame(sheet_df).to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute automatic metrics for LLM outputs.")
    parser.add_argument("--input-dir", default="llm output")
    parser.add_argument("--output-dir", default="automatic metrics results")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names (default: built-in list).")
    parser.add_argument(
        "--camembert-model",
        default="dangvantuan/sentence-camembert-base",
        help="SentenceTransformer model name for French similarity.",
    )
    args = parser.parse_args()

    datasets = DEFAULT_DATASETS
    run(datasets=datasets, input_dir=args.input_dir, output_dir=args.output_dir, camembert_model_name=args.camembert_model)


if __name__ == "__main__":
    main()