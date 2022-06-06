from pytorch_lightning.callbacks import ModelCheckpoint
import os

folder = "checkpoints"


def valid_loss_callback(tag):
    return ModelCheckpoint(
        monitor="valid_loss",
        dirpath=os.path.join(os.getcwd(), folder, tag, "loss"),
        filename="{epoch:02d}-{valid_loss:.2f}",
        save_top_k=5,
        mode="min"
    )


def valid_acc_callback(tag, mode, train_ops, train_ops_mat_only, train_ops_with_boundary, seed):
    mode = "_mode" + str(mode)
    if train_ops:
        mode = mode + "_ops"
    if train_ops_mat_only:
        mode = mode + "_mat"
    if train_ops_with_boundary:
        mode = mode + "_bdy"
    print("save max accuracy")
    return ModelCheckpoint(
        monitor="valid_acc_epoch",
        dirpath=os.path.join(os.getcwd(), folder, tag + mode, "accuracy"),
        filename="{epoch:02d}-{valid_acc_epoch:.2f}--" + str(seed),
        save_top_k=1,
        mode="max"
    )


def last_callback(tag):
    return ModelCheckpoint(
        # save_last=True,
        save_top_k=-1,
        dirpath=os.path.join(os.getcwd(), folder, tag, "last"),
        filename= "{epoch:02d}"
    )


def mean_iu_callback(tag):
    return ModelCheckpoint(
        monitor="valid_meaniu",
        dirpath=os.path.join(os.getcwd(), folder, tag, "meaniu"),
        filename="minc-{epoch:02d}-{valid_meaniu:.2f}",
        save_top_k=5,
        mode="max"
    )