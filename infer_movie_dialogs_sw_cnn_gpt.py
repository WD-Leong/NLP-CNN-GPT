import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_cnn_gpt_keras as gpt

# Model Parameters. #
seq_length = 31
num_heads  = 4
num_layers = 3

kernel_sz = 3
prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "TF_Models/dialogue_sw_cnn_gpt"
train_loss_file = "train_loss_dialogue_sw_cnn_gpt.csv"

# Load the data. #
tmp_pkl_file = "../../Data/movie_dialogs/"
tmp_pkl_file += "movie_dialogues_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_2_idx)
print("Vocabulary Size:", str(vocab_size) + ".")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Build the GPT. #
print("Building the GPT Keras Model.")
start_time = time.time()

gpt_model = gpt.GPTDecoder(
    num_layers, num_heads, 
    hidden_size, ffwd_size, vocab_size, seq_length, 
    ker_sz=kernel_sz, rate1=0.0, rate2=1.0-prob_keep)
gpt_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time() - start_time) / 60
print("GPT Keras Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

train_loss_df = pd.read_csv(train_loss_file)
train_loss_list = [tuple(
    train_loss_df.iloc[x].values) \
    for x in range(len(train_loss_df))]

# GPT model inference. #
n_iter = ckpt.step.numpy().astype(np.int32)

print("-" * 50)
print("GPT model inference", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Update the neural network's weights. #
while True:
    tmp_phrase = input("Enter prompt: ")
    tmp_phrase = tmp_phrase.strip().lower()
    if tmp_phrase == "":
        break
    else:
        tmp_encode = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword_2_idx)
        tmp_encode += [SOS_token]

        n_tokens  = len(tmp_encode)
        tmp_array = np.array(tmp_encode).reshape((1, -1))
        tmp_infer = gpt_model.infer(tmp_array)
        array_ids = tmp_infer[0].numpy()

        gen_phrase = bpe.bp_decode(
            array_ids, idx_2_subword)
        gen_output = bpe.bp_decode(
            array_ids[(n_tokens-1):], idx_2_subword)
        
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        gen_output = " ".join(
            gen_output).replace("<", "").replace(">", "")
        
        print("")
        print("Input Phrase:")
        print(tmp_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("Generated Response:")
        print(gen_output)
        print("-" * 50)
