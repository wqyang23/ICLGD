import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
# from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

model_path = "./models/llama-3.2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


dataset_name = ['hotel', 'sst2', 'sst5', 'subj', 'agnews']
dataset = load_from_disk(f'./datasets/{dataset_name}')


def get_top_10_samples_by_label(dataset):
    train_data = dataset['train']
    label_to_samples = {}

    for sample in train_data:
        label = sample['label']
        if label not in label_to_samples:
            label_to_samples[label] = []
        if len(label_to_samples[label]) < 10:
            label_to_samples[label].append(sample['text'])

    processed_data = []
    for label, texts in label_to_samples.items():
        for text in texts:
            processed_data.append({'text': text, 'label': label})

    return processed_data


processed_train_data = get_top_10_samples_by_label(dataset)

train_dataset = Dataset.from_dict(processed_train_data)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


train_dataset = train_dataset.map(tokenize_function, batched=True)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

trainer.save_model("./fine_tuned_model_lora")
