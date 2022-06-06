import torch
# from fvcore.nn import FlopCountAnalysis, flop_count_table
from pytorch_lightning import Trainer
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything
import os
# from time import time

from torchtools.dataloaders.lmd import LMDSegLoader
from torchtools.dataloaders.opsurface import OPSLoader
from torchtools.utils.arg_parser import ArgParser
from torchtools.experiments.sota_segmenter import SOTASegmenter
from torchtools.utils.callbacks import valid_acc_callback
# from ptflops import get_model_complexity_info

"""
    Training mode: 
    99 - Train with LMD, with no pre-training.
    98 - Train with LMD, with pre-training.
"""


def main():
    # warnings.filterwarnings('ignore')
    # Start training from scratch
    seed_everything(args.seed)
    # logger = TensorBoardLogger("./results", name=args.tag)
    # for HPC training
    if not args.debug:
        trainer = Trainer(progress_bar_refresh_rate=20, log_every_n_steps=20, flush_logs_every_n_steps=800,
                          max_epochs=args.epochs,
                          gpus=args.gpus,
                          # num_nodes=args.num_nodes,
                          # accelerator='ddp',
                          # replace_sampler_ddp=True,
                          callbacks=[valid_acc_callback(args.tag, args.mode, args.train_ops, args.train_ops_mat_only, args.train_ops_with_boundary, args.seed)],
                          accumulate_grad_batches=4,
                          # stochastic_weight_avg=True,
                          # plugins=DDPPlugin(find_unused_parameters=True),
                          logger=None)
    else:
        # for CPU training
        trainer = Trainer(progress_bar_refresh_rate=1, log_every_n_steps=1, flush_logs_every_n_steps=1,
                          max_epochs=args.epochs, replace_sampler_ddp=False, accelerator='ddp_cpu', num_processes=1,
                          callbacks=[valid_acc_callback(args.tag, args.mode, args.train_ops, args.train_ops_mat_only, args.train_ops_with_boundary, args.seed)],
                          # stochastic_weight_avg=True,
                          # plugins=DDPPlugin(find_unused_parameters=True),
                          logger=None)

    # Parse the argements
    # json_data = arg_parser.json_data
    # load trained encoder
    if args.train_ops:
        dm = OPSLoader(args.data_root, batch_size=args.batch_size)
    else:
        dm = LMDSegLoader(args.data_root, batch_size=args.batch_size, split=args.split)
    # resume
    if args.resume or args.test or args.pre_train or args.testindoor:
        # Parse the argements
        ckpt_path = args.resume or args.test or args.pre_train or args.testindoor
        mode = "_mode" + str(args.mode)
        if args.train_ops:
            mode = mode + "_ops"
        if args.train_ops_mat_only:
            mode = mode + "_mat"
        if args.train_ops_with_boundary:
            mode = mode + "_bdy"
        ckpt_path = os.path.join(os.getcwd(), "checkpoints", args.tag+mode, ckpt_path)
        if torch.cuda.is_available():
            checkpoint = torch.load(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = SOTASegmenter(checkpoint, mode=args.mode, net=args.tag, saveoutput=True if args.testindoor else None, crf=True if args.crf else None, crftest=True if args.crftest else None,
                              train_ops=True if args.train_ops else False, train_ops_mat_only=True if args.train_ops_mat_only else False,
                              train_with_boundary_loss=True if args.train_ops_with_boundary else False, ckpt_path=args.tag+mode, uncertainty=args.uncertainty)
        if not args.pre_train:
            trainer.global_step = checkpoint["global_step"]
            trainer.current_epoch = checkpoint["epoch"]
    else:
        model = SOTASegmenter(mode=args.mode, net=args.tag, crf=True if args.crf else None, crftest=True if args.crftest else None,
                              train_ops=True if args.train_ops else False, train_ops_mat_only=True if args.train_ops_mat_only else False,
                              train_with_boundary_loss=True if args.train_ops_with_boundary else False, ckpt_path=None, uncertainty=args.uncertainty)

    # Calculate FPS
    # inputs = torch.randn(1, 3, 512, 512)
    # model = model.net
    # model.eval()
    # start_time = time()
    # iter=100
    # with torch.no_grad():
    #     for i in range(iter):
    #         model(inputs)
    # end_time = time()
    # print("{} seconds used to calculate {} forward paths, FPS={}".format(end_time-start_time, iter, iter/(end_time-start_time)))
    # exit(0)
    # print(flop_count_table(FlopCountAnalysis(model, inputs), max_depth=1))

    if not args.test and not args.testindoor and not args.infer:
        trainer.fit(model, dm)

    # run test set
    if args.test or args.testindoor or args.infer:
        if args.testindoor:
            print("indoor image will be saved")
            model.saveoutput = True
            trainer.test(model, test_dataloaders=dm.test_indoor_dataloader(), ckpt_path=None)
        elif args.infer:
            print("infer images in folder {}".format(args.infer))
            model.infer(dataloader=dm.infer_dataloader(args.infer))

        else:
            trainer.test(model, datamodule=dm, ckpt_path=None)


if __name__ == '__main__':
    arg_parser = ArgParser()
    args = arg_parser.args
    main()
