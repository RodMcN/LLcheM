from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter

class Logger:
    def __init__(self, tensorbard_dir=None) -> None:
        if tensorbard_dir:
            self.tensorbard_writer = TBSummaryWriter(log_dir=tensorbard_dir)
        else:
            self.tensorbard_writer = None
    
    def add_scalar(self, tag, scalar_value, global_step=None):
        if self.tensorbard_writer is not None:
            self.tensorbard_writer.add_scalar(tag, scalar_value, global_step)