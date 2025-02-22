import os
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import AutoTokenizer
import evaluate

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


# Key Fix 1: Modified callback to handle metrics computation
class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.best_rouge = 0.0

    def on_evaluate(self, args, state, control, **kwargs):
        # Get metrics from last evaluation
        metrics = state.log_history[-1]
        if "eval_rougeL" in metrics:
            current_rouge = metrics["eval_rougeL"]
            if current_rouge > self.best_rouge:
                self.best_rouge = current_rouge
                kwargs["model"].save_pretrained("./best_model")
                kwargs["tokenizer"].save_pretrained("./best_model")
                print(f"🔥 New best model saved with ROUGE-L: {current_rouge:.2f}")


def simulate_html(article):
    # Split the article into paragraphs
    paragraphs = article.split("\n")
    # Wrap each paragraph in <p> tags
    html_content = "\n".join([f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()])
    return html_content


# Key Fix 2: Fixed preprocessing to match dataset columns
def preprocess_function(examples):
    inputs = []
    for text in examples["input"]:  # Match actual column name from your dataset
        html_content = simulate_html(text)
        inputs.append(f"question: Summarize the article. context: {html_content}")
    return {"inputs": inputs, "targets": examples["summary"]}


# Key Fix 3: Proper tokenizer handling
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    inputs = tokenizer(
        examples["inputs"], max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["targets"], max_length=128, truncation=True, padding="max_length"
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"],
    }


# Key Fix 4: Added metrics computation to trainer
def compute_metrics(eval_pred):
    rouge = evaluate.load("rouge")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )


def train_model(dataset, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Key Fix 5: Correct training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-cnn-summarizer",
        evaluation_strategy="epoch",  # Fixed typo from 'eval_strategy'
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        save_strategy="no",
        report_to="none",
    )

    split_dataset = dataset["train"].train_test_split(test_size=0.1)

    # Key Fix 6: Added metrics computation to trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Added metrics computation
        callbacks=[SaveBestModelCallback()],  # Simplified callback
    )

    trainer.train()


if __name__ == "__main__":
    # Verify dataset structure matches your preprocessing
    ds = load_dataset("kritsadaK/EDGAR-CORPUS-Financial-Summarization")

    # Check actual column names in the dataset
    print(ds["train"][0].keys())  # Verify columns match preprocessing

    # Preprocess dataset
    tokenized_ds = ds.map(
        preprocess_function, batched=True, remove_columns=ds["train"].column_names
    ).map(tokenize_function, batched=True)

    train_model(tokenized_ds, "google/flan-t5-small")
