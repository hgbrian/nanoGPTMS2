# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-ms2'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 50 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'ms2'

dataset = 'ms2'
gradient_accumulation_steps = 1
batch_size = 24
block_size = 128 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 1024
dropout = 0.2
n_bottleneck = 64

wandb_run_name = f"ms2-{n_bottleneck}_{n_layer}_{n_layer}_{n_embd}"

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 100_000_000 # just interrupt
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
