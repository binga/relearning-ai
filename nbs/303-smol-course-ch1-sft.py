from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
import trackio as wandb
from trl import SFTTrainer, SFTConfig

wandb.init(project="my-sft", name="sft-runs")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

dataset = load_dataset("HuggingFaceTB/smoltalk2", "SFT")
data

config = SFTConfig(
    output_dir="./models",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    max_steps=20,
    logging_steps=2,
    report_to="trackio"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=data['train'],
    args=config
)

trainer.train()

