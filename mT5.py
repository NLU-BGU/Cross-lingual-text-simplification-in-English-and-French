import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import argparse
import os
from typing import Iterable, Optional


def train_model(
    model_type: str,
    model_name: str,
    train_data: pd.DataFrame,
    eval_data: Optional[pd.DataFrame] = None,
    seq_length: int = 256,
    batch_size: int = 8,
    acc_steps: int = 4,
    epochs: int = 1,
    do_sample: bool = True,
    top_k: Optional[int] = None,
    top_p: float = 0.95,
    num_beams: int = 1,
    seed: int = 45,
) -> T5Model:
    """Train a SimpleTransformers mT5 model.

    Args:
        model_type: Model type (e.g., "mt5").
        model_name: HF model name (e.g., "google/mt5-small").
        train_data: Training dataset DataFrame.
        eval_data: Optional evaluation dataset DataFrame.
        seq_length: Max sequence length for inputs.
        batch_size: Training batch size.
        acc_steps: Gradient accumulation steps.
        epochs: Number of training epochs.
        do_sample: Whether to sample during generation.
        top_k: Top-k sampling value.
        top_p: Top-p (nucleus) sampling value.
        num_beams: Beam count for beam search.
        seed: Random seed.

    Returns:
        Trained `mT5 Model`.
    """
    model_args = T5Args()
    model_args.max_seq_length = seq_length
    model_args.max_length = 512
    model_args.train_batch_size = batch_size
    model_args.gradient_accumulation_steps = acc_steps
    model_args.gradient_checkpointing = True
    model_args.eval_batch_size = batch_size
    model_args.num_train_epochs = epochs
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 300
    model_args.use_multiprocessing = False
    model_args.fp16 = False
    model_args.save_steps = -1
    model_args.save_eval_checkpoints = False
    model_args.no_cache = True
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.preprocess_inputs = True
    model_args.optimizer = "AdamW"
    model_args.num_return_sequences = 1
    model_args.lazy_loading = False
    model_args.no_save = True
    model_args.do_sample = do_sample
    model_args.top_k = top_k
    model_args.top_p = top_p
    model_args.num_beams = num_beams
    model_args.save_model_every_epoch = False
    model_args.manual_seed = seed 
    model_args.use_cuda=False
    model_args.process_count = 1 
    model = T5Model(model_type, model_name, args=model_args, use_cuda=False)

    model.train_model(train_data, eval_data=eval_data)
    return model 




def training_loop(
    training_data: pd.DataFrame,
    validation_data: Optional[pd.DataFrame],
    model_name: str = "google/mt5-small",
    epochs: int = 1,
    do_sample: bool = True,
    top_p: float = 0.95,
) -> T5Model:
    """Train an MT5 model with a simple preset configuration."""
    model_type = "mt5"
    return train_model(
        model_type=model_type,
        model_name=model_name,
        train_data=training_data,
        eval_data=validation_data,
        do_sample=do_sample,
        top_p=top_p,
        epochs=epochs,
    )

DEFAULT_CORPORA = ["clear", "MultiCochrane", "asset", "wikilargefr", "wikiauto"]


def _parse_csv_list(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def run(
    corpora: Iterable[str],
    training_dir: str = "mt5-training",
    outputs_dir: str = os.path.join("mt5-training", "outputs"),
    model_name: str = "google/mt5-small",
    epochs: int = 1,
) -> None:
    """Train per-corpus MT5 models and write prediction spreadsheets."""
    os.makedirs(outputs_dir, exist_ok=True)

    for corp in corpora:
        training_path = os.path.join(training_dir, f"{corp}- training set.xlsx")
        training_set = pd.read_excel(training_path)

        if corp != "asset":
            validation_path = os.path.join(training_dir, f"{corp}- validation set.xlsx")
            validation_set = pd.read_excel(validation_path)
        else:
            validation_set = None

        model = training_loop(training_set, validation_set, model_name=model_name, epochs=epochs)

        test_path = os.path.join(training_dir, f"{corp}.xlsx")
        test_set = pd.read_excel(test_path)
        complex_cols = [col for col in test_set.columns if "Complex" in str(col)]
        if not complex_cols:
            raise ValueError(f"No column containing 'Complex' found in {test_path}")
        complex_col = str(complex_cols[0])

        results = model.predict(test_set[complex_col].astype(str).tolist())
        test_set["Predictions-mt5"] = results

        out_path = os.path.join(outputs_dir, f"{corp} mt5 results.xlsx")
        test_set.to_excel(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MT5 and generate predictions per corpus.")
    parser.add_argument("--training-dir", default="mt5-training")
    parser.add_argument("--outputs-dir", default=os.path.join("mt5-training", "outputs"))
    parser.add_argument("--corpora", default=None, help="Comma-separated corpora (default: built-in list).")
    parser.add_argument("--model-name", default="google/mt5-small")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    corpora = _parse_csv_list(args.corpora) or DEFAULT_CORPORA
    run(
        corpora=corpora,
        training_dir=args.training_dir,
        outputs_dir=args.outputs_dir,
        model_name=args.model_name,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
