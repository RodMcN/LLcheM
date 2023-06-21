import config

if config.n_gpu <= 1:
    from train import train
    print("Training on Single GPU")
    train()

else:
    from torch.distributed.run import main as launch_distributed_training
    print(f"Training on {config.n_gpu} GPUs")
    dist_args = ["--standalone", 
                 f"--nproc_per_node={config.n_gpu}", 
                 "--max_restarts" , "0",
                 "train.py",]
    launch_distributed_training(dist_args)