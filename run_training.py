from config import Config, local_data

if Config.log_wandb_:
    import os, wandb
    os.system(f"wandb login {local_data.wandb_key}")

    wandb.init(
        project="LLcheM",
        name=Config.expt_name,
        config=Config.export_dict()
    )  

if Config.n_gpu <= 1:
    from train import train
    print("Training on Single GPU")
    train()

else:
    from torch.distributed.run import main as launch_distributed_training
    print(f"Training on {Config.n_gpu} GPUs")
    dist_args = ["--standalone", 
                 f"--nproc_per_node={Config.n_gpu}", 
                 "--max_restarts" , "0",
                 "train.py",]
    launch_distributed_training(dist_args)