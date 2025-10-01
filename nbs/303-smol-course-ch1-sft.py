# /// script
# dependencies = [
#     "datasets>=3.6.0",
#     "dspy>=3.0.3",
#     "einops>=0.8.1",
#     "joblib>=1.5.1",
#     "lovely-tensors>=0.1.18",
#     "matplotlib>=3.10.3",
#     "numpy>=2.3.1",
#     "peft>=0.17.1",
#     "torch>=2.7.1",
#     "torchinfo>=1.8.0",
#     "trackio>=0.2.5",
#     "transformers>=4.53.1",
#     "trl>=0.23.0",
# ]
# ///


from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
import trackio
from trl import SFTTrainer, SFTConfig

trackio.init(project="smol-course", name="sft-runs")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

dataset = load_dataset("roneneldan/TinyStories")

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
    train_dataset=dataset['train'].select(range(100)),
    args=config,
)

trainer.train()