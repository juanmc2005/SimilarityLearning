from core.base import TrainingListener, TestListener


class ScreenProgressLogger:

    def __init__(self, interval, n_batch):
        super(ScreenProgressLogger, self).__init__()
        self.interval = interval
        self.n_batch = n_batch
        self.train_log_ft = "Train Epoch: {epoch} [{progress}%]\tLoss: {loss:.6f}"
        self.test_log_ft = "Testing [{progress}%]"
        self.last_log = -1

    def _progress(self, i):
        progress = int(100. * (i + 1) / self.n_batch)
        should_log = progress > self.last_log and progress % self.interval == 0
        return progress, should_log

    def restart(self):
        self.last_log = -1

    def on_train_batch(self, i, epoch, loss):
        progress, should_log = self._progress(i)
        if should_log:
            self.last_log = progress
            print(self.train_log_ft.format(epoch=epoch, progress=progress, loss=loss))

    def on_test_batch(self, i):
        progress, should_log = self._progress(i)
        if should_log:
            self.last_log = progress
            print(self.test_log_ft.format(progress=progress))


class TrainLogger(TrainingListener):

    def __init__(self, interval, nbatches, loss_log_path: str):
        super(TrainLogger, self).__init__()
        self.nbatches = nbatches
        self.logger = ScreenProgressLogger(interval, nbatches)
        self.total_loss = 0
        self.log_file_path = loss_log_path
        open(loss_log_path, 'w').close()

    def on_before_epoch(self, epoch):
        self.total_loss = 0
        self.logger.restart()

    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.total_loss += loss
        self.logger.on_train_batch(ibatch, epoch, loss)

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        mean_loss = self.total_loss / self.nbatches
        print(f"[Epoch {epoch} finished. Mean Loss: {mean_loss:.6f}]")
        with open(self.log_file_path, 'a') as logfile:
            logfile.write(f"{mean_loss}\n")


class TestLogger(TestListener):

    def __init__(self, interval, n_batch):
        super(TestLogger, self).__init__()
        self.logger = ScreenProgressLogger(interval, n_batch)

    def on_before_test(self):
        self.logger.restart()

    def on_batch_tested(self, ibatch, feat):
        self.logger.on_test_batch(ibatch)


class MetricFileLogger(TestListener):

    def __init__(self, log_path: str):
        self.log_file_path = log_path
        open(log_path, 'w').close()

    def on_after_test(self, epoch, feat_test, y_test, metric_value):
        with open(self.log_file_path, 'a') as logfile:
            logfile.write(f"{metric_value}\n")


class HeaderPrinter(TrainingListener):

    def __init__(self):
        self.current_header_size = 0

    def on_before_epoch(self, epoch):
        header = f"--------------- Epoch {epoch:02d} ---------------"
        self.current_header_size = len(header)
        print(header)

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        print('-' * self.current_header_size)
