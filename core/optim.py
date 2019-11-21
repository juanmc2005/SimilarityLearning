

class Optimizer:

    def __init__(self, optimizers, schedulers):
        super(Optimizer, self).__init__()
        self.OPTIM_KEY = 'optimizers'
        self.SCHED_KEY = 'schedulers'
        self.optimizers = optimizers
        self.schedulers = schedulers

    def state_dict(self):
        return {
            self.OPTIM_KEY: [op.state_dict() for op in self.optimizers],
            self.SCHED_KEY: [s.state_dict() for s in self.schedulers]
        }

    def load_state_dict(self, checkpoint):
        if self.OPTIM_KEY in checkpoint:
            for i, op in enumerate(self.optimizers):
                op.load_state_dict(checkpoint[self.OPTIM_KEY][i])
        if self.SCHED_KEY in checkpoint:
            for i, s in enumerate(self.schedulers):
                s.load_state_dict(checkpoint[self.SCHED_KEY][i])

    def scheduler_step(self, metric: float = None):
        for s in self.schedulers:
            if metric is None:
                s.step()
            else:
                s.step(metric)

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step(self):
        for o in self.optimizers:
            o.step()

    def lrs(self):
        return [o.state_dict()['param_groups'][0]['lr'] for o in self.optimizers]
