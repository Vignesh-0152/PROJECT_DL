from tensorflow.keras.callbacks import Callback
import sys

class LiveTerminalOutput(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or{}
        loss = logs.get('loss',0)
        acc = logs.get('accuracy',0)
        precision = logs.get('precision',0)
        recall = logs.get('recall',0)
        auc = logs.get('auc',0)

        msg = (f"Batch: {batch+1} - "
            f"loss: {loss:.4f} - "
            f"accuracy: {acc:.4f} - "
            f"precision: {precision:.6f} - "
            f"recall: {recall:.4f} - "
            f"auc: {auc:.4f}")

        # Use sys.stdout.write and flush to overwrite the same line
        sys.stdout.write('\r' + msg)
        sys.stdout.flush()