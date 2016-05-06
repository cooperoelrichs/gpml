from keras.callbacks import EarlyStopping
import warnings


class BestOnlyEarlyStopping(EarlyStopping):
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: early stopping' % (epoch))
                    print('Setting weights to those from epoch %i.'
                          % self.best_epoch)
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
            self.wait += 1
