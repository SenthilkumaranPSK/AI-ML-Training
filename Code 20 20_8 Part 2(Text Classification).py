import streamlit as st
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

st.title("Custom Text Classification with Hugging Face")

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV with columns: text,label)", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.dataframe(df)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)

    label_names = list(df["label"].unique())
    label2id = {l: i for i, l in enumerate(label_names)}
    id2label = {i: l for i, l in enumerate(label_names)}

    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    dataset = dataset.map(encode_labels)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")

    num_labels = len(label_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",    
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    )

    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer.train()

        st.success("Training complete!")

        model.save_pretrained("./saved_model")
        tokenizer.save_pretrained("./saved_model")

if os.path.exists("./saved_model"):
    st.write("### Try Real-time Predictions")
    user_input = st.text_input("Enter a sentence:")
    if user_input:
        clf = pipeline(
            "text-classification",
            model="./saved_model",
            tokenizer="./saved_model"
        )
        prediction = clf(user_input)[0]
        st.write(
            f"**Prediction:** {prediction['label']} "
            f"(score: {prediction['score']:.4f})"
        )
