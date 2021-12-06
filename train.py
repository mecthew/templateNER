import os

import pandas as pd
import logging
import argparse
from seq2seq_model import Seq2SeqModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="conll2003",
                    help="dataset name")
parser.add_argument("--encoder_decoder_type", type=str, default="bart")
parser.add_argument("--pretrain_path", type=str, default="",
                    help="pretrained model path")
parser.add_argument("--reprocess_input_data", type=int, default=1)
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=100)
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=4e-5)
parser.add_argument("--target_max_length", type=int, default=25)
parser.add_argument("--manual_seed", type=int, default=4)
parser.add_argument("--save_model_steps", type=int, default=11898)
parser.add_argument("--wandb_project", type=str, default="templateNER")

args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

train_data = pd.read_csv(f"./data/{args.dataset}/train.csv", sep=',').values.tolist()
train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = pd.read_csv(f"./data/{args.dataset}/dev.csv", sep=',').values.tolist()
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])
model_name = args.pretrain_path.replace('\\', '/').rsplit('/')[-1]

model_args = {
    "reprocess_input_data": bool(args.reprocess_input_data),
    "overwrite_output_dir": True,
    "max_seq_length": args.max_seq_length,
    "train_batch_size": args.train_batch_size,
    "num_train_epochs": args.num_train_epochs,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "multiprocessing_chunksize": 4,
    "max_length": args.target_max_length,
    "manual_seed": args.manual_seed,
    "save_steps": args.save_model_steps,
    "gradient_accumulation_steps": 1,
    "output_dir": f"./outputs_{args.dataset}_{model_name}",
    "best_model_dir": f"./outputs_{args.dataset}_{model_name}/best_model"
}

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type=args.encoder_decoder_type,
    encoder_decoder_name=args.pretrain_path,
    args=model_args,
    use_cuda=True,
)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction

print(model.predict([
                        "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday."]))
