import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchtools.models.dpglt.helper import create_model_load_weights, get_optimizer
from torchtools.models.dpglt.utils.loss import FocalLoss, BoundaryLoss
from torchtools.metrics.segment_scores import ConfusionMatrix
from torchtools.utils.segment_helper import confusion_matrix
from torchtools.metrics.segment_scores import Loss
from ..dataloaders.lmd import patch2global
from torchtools.optimizers.poly_lr_scheduler import PolyLRScheduler
import numpy as np
from PIL import Image
from torch.nn import CrossEntropyLoss


def get_scores(confmat, istest=False, save_conf=False, conf_name="default.npy"):
    # accuracy across all categories
    acc = torch.diag(confmat).sum() / confmat.sum()

    # accuracy by averaging all category accuracies
    acc_cls = torch.diag(confmat) / confmat.sum(axis=1)
    acc_cls[torch.isinf(acc_cls)] = float('nan')
    if istest:
        print("mean class accuracy: {}".format(acc_cls))
    acc_cls[torch.isnan(acc_cls)] = 0
    acc_cls_mean = torch.mean(acc_cls)

    if istest:
        # iou score
        intersection = torch.diag(confmat)
        union = confmat.sum(0) + confmat.sum(1) - intersection
        scores = intersection.float() / union.float()
        scores[torch.isinf(scores)] = float('nan')
        # valid_idx = [1, 3, 4, 6, 7, 8, 9, 10, 15]
        # print(scores)
        print("iou score:{}".format(torch.mean(scores)))

    if save_conf:
        torch.save(confmat, "{}.pt".format(conf_name))
    return acc, acc_cls_mean


def segment2rgb(output, mask=False, ops=False):
    if ops:
        pass
    else:
        color_plate = {0: [119, 17, 17], 1: [202, 198, 144], 2: [186, 200, 238], 3: [124, 143, 166], 4: [89, 125, 49],
                   5: [16, 68, 16], 6: [187, 129, 156], 7: [208, 206, 72], 8: [98, 39, 69], 9: [102, 102, 102],
                   10: [76, 74, 95], 11: [16, 16, 68], 12: [68, 65, 38], 13: [117, 214, 70], 14: [221, 67, 72],
                   15: [92, 133, 119]}
    if not mask:
        output = output.argmax(dim=1)
    output = output.squeeze().cpu()
    rgbmask = np.zeros([output.size()[0], output.size()[1], 3], dtype=np.uint8)
    for i in range(16):
        rgbmask[output == i] = color_plate[i]

    return rgbmask


def resize_img_tensors(merged_tensors, segments_tensor):
    assert len(merged_tensors) == len(segments_tensor), "number of images does not match with segments"
    resized_tensors = []
    for idx in range(len(merged_tensors)):
        if merged_tensors[idx].size()[-2:] != segments_tensor[idx].size()[-2:]:
            resized_tensors.append(F.interpolate(merged_tensors[idx], size=(segments_tensor[idx].size(-2), segments_tensor[idx].size(-1)), mode="nearest"))
        else:
            resized_tensors.append(merged_tensors[idx])
    return resized_tensors


class SOTASegmenter(pl.LightningModule):
    def __init__(self, ckpt=None, n_class=16, mode=1, net="efficientnet", saveoutput=False, crf=None, crftest=None, train_ops=False, train_ops_mat_only=False, train_with_boundary_loss=False, ckpt_path=None, uncertainty=False):
        super(SOTASegmenter, self).__init__()
        self.n_class = n_class  # default train LMD
        self.ckpt = ckpt
        self.mode = mode
        self.net = create_model_load_weights(net, self.n_class, img_size=(512, 512), mode=self.mode, crf=crf, train_ops=train_ops, train_ops_mat_only=train_ops_mat_only)
        self.netname = net
        self.automatic_optimization = True
        self.saveoutput = saveoutput
        self.crf = crf
        self.crftest = crftest
        self.train_ops = train_ops
        self.train_ops_mat_only = train_ops_mat_only
        self.train_with_boundary_loss = train_with_boundary_loss
        self.ckpt_path = ckpt_path        

        if uncertainty:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.net.segmentation_head.parameters():
                param.requires_grad = True
            print("now only train the segmentation head")

        if self.ckpt is not None:
            if self.crf is not None:
                if self.crftest is not None:
                    self.net.load_state_dict(self.ckpt["state_dict"], strict=True)
                else:
                    self.net.previous_model.load_state_dict(self.ckpt["state_dict"], strict=True)
            else:
                self.net.load_state_dict(self.ckpt["state_dict"], strict=True)
            print("Loaded trained DPGLT model")

        self.optimizer = None
        self.scheduler = None
        if not self.train_ops:
            self.criterion = FocalLoss(gamma=3)
            self.confmat_train = ConfusionMatrix(num_classes=self.n_class)
            self.confmat_valid = ConfusionMatrix(num_classes=self.n_class)
            self.confmat_test = ConfusionMatrix(num_classes=self.n_class)
        else:
            self.material_criterion = FocalLoss(gamma=3)
            if train_with_boundary_loss:
                self.material_boundary_criterion = BoundaryLoss()
            # self.object_criterion = FocalLoss(gamma=3)
            # self.scene_criterion = CrossEntropyLoss(ignore_index=-1)

            self.confmat_train = ConfusionMatrix(num_classes=27)
            self.confmat_valid = ConfusionMatrix(num_classes=27)
            self.confmat_test = ConfusionMatrix(num_classes=27)

        self.train_loss = Loss()
        self.valid_loss = Loss()
        self.test_loss = Loss()

        # define patch size
        self.patch_size = 512

    def forward(self, batch, training=True):
        # batch: two lists, one images, one segments
        if training:
            images, segments, images_255, names = batch
            if self.crf is not None:
                images_rgb = images_255 - torch.tensor([122.675, 116.669, 104.008],
                                                                         device=images_255[0].device).view(
                    1, -1, 1, 1)
                preds = self.net(images, images_rgb)
            else:
                preds = self.net(images)
            if not self.train_ops:
                loss, count = self.criterion(preds, segments, softmax=True)
                confmat = confusion_matrix(preds, segments, self.n_class)
            else:
                if not self.train_ops_mat_only:
                    material_mask_preds, object_mask_preds, scene_label_preds = preds
                else:
                    material_mask_preds = preds
                material_segments, object_segments, scene_labels = segments

                loss, count = self.material_criterion(material_mask_preds, material_segments, softmax=True)
                if self.train_with_boundary_loss:
                    loss_boundary, _ = self.material_boundary_criterion(material_mask_preds, material_segments,
                                                                        softmax=True)
                confmat = confusion_matrix(material_mask_preds, material_segments, 27)

        else:
            splitted_tensors, count_mask_tensors, patch_count, resized_size, segments_tensor, images_255, names = batch

            if self.crf is not None:
                images_rgb = images_255 - torch.tensor([122.675, 116.669, 104.008],device=images_255.device).view(
                    1, -1, 1, 1)
                preds = self.net(splitted_tensors, images_rgb)
            else:
                preds = self.net(splitted_tensors)
            # back to 4 times upsampling.
            # preds = F.interpolate(preds, size=splitted_tensors.size()[-2:], mode="nearest")
            if not self.train_ops:
                merged_tensors = patch2global(preds, count_mask_tensors, patch_count, resized_size)
                preds = resize_img_tensors(merged_tensors, segments_tensor)
                segments = segments_tensor

                loss, count = self.criterion(preds, segments, softmax=True)
                confmat = confusion_matrix(preds, segments, self.n_class)
                if self.saveoutput:
                    assert len(preds) == 1, "only support single batch test"
                    img = Image.fromarray(np.asarray(segment2rgb(preds[0])))
                    if not os.path.isdir(os.path.join(os.getcwd(), "output", str(self.netname)+"_"+str(self.mode))):
                        os.makedirs(os.path.join(os.getcwd(), "output", str(self.netname)+"_"+str(self.mode)))

                    img.save(os.path.join(os.getcwd(), "output", str(self.netname)+"_"+str(self.mode), "{}.png").format(names[0]))

            else:
                if not self.train_ops_mat_only:
                    material_mask_preds, object_mask_preds, scene_label_preds = preds

                else:
                    material_mask_preds = preds

                material_segments, object_segments, scene_labels = segments_tensor

                merged_material_tensors = resize_img_tensors(patch2global(material_mask_preds, count_mask_tensors, patch_count, resized_size), material_segments)
                # merged_object_tensors = resize_img_tensors(patch2global(object_mask_preds, count_mask_tensors, patch_count, resized_size), object_segments)

                loss, count = self.material_criterion(merged_material_tensors, material_segments, softmax=True)
                if self.train_with_boundary_loss:
                    loss_boundary, _ = self.material_boundary_criterion(merged_material_tensors, material_segments, softmax=True)
                confmat = confusion_matrix(merged_material_tensors, material_segments, 27)

        if not self.train_ops:
            return loss, count, confmat
        else:
            if self.train_ops_mat_only:
                if self.train_with_boundary_loss:
                    return 0.5*loss+0.5*loss_boundary, count, confmat
                else:
                    return loss, count, confmat
            else:
                pass
                # implement with object, scene loss

    def infer(self, dataloader):
        for idx, batch in enumerate(dataloader):
            splitted_tensors, count_mask_tensors, patch_count, resized_size, segments_tensor, images_255, names = batch
            if self.crf is not None:
                images_rgb = images_255 - torch.tensor([122.675, 116.669, 104.008],device=images_255.device).view(
                    1, -1, 1, 1)
                preds = self.net(splitted_tensors, images_rgb)
            else:
                preds = self.net(splitted_tensors)
            merged_tensors = patch2global(preds, count_mask_tensors, patch_count, resized_size)
            preds = resize_img_tensors(merged_tensors, segments_tensor)
            img_rgb = Image.fromarray(np.asarray(segment2rgb(preds[0])))

            output = preds[0].argmax(dim=1).squeeze().cpu()
            if not os.path.isdir(os.path.join(os.getcwd(), "output")):
                os.makedirs(os.path.join(os.getcwd(), "output"))

            if not os.path.isdir(os.path.join(os.getcwd(), "output", names[0].split("/")[0])):
                os.makedirs(os.path.join(os.getcwd(), "output", names[0].split("/")[0]))

            img = Image.fromarray(np.asarray(output).astype(np.uint8), "L")
            img.save(os.path.join(os.getcwd(), "output", "{}.png").format(names[0]))
            img_rgb.save(os.path.join(os.getcwd(), "output", "{}.png").format(names[0]+"rgb"))

    def training_step(self, batch, batch_idx):
        loss, count, confmat = self.forward(batch, training=True)
        self.confmat_train(confmat)  # confusion matrix of patches.
        self.log("train_loss", self.train_loss(loss, count), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
                 logger=False)
        if self.scheduler._decide_stage() == 0:
            self.scheduler.step(self.trainer.current_epoch)

        return loss

    def training_epoch_end(self, outputs):
        if self.scheduler._decide_stage() == 1:
            self.scheduler.step(self.trainer.current_epoch)
        confmat = self.confmat_train.compute()
        self.confmat_train.reset()

        acc, acc_mean = get_scores(confmat)
        # calculate valid_acc
        self.log("train_acc_epoch", acc, prog_bar=True)
        self.log("train_acc_mean_epoch", acc_mean, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, count, confmat = self.forward(batch, training=False)

        self.confmat_valid(confmat)  # confusion matrix of patches.
        self.log("valid_loss", self.valid_loss(loss, count), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
                 logger=False)

    def validation_epoch_end(self, outputs):
        confmat = self.confmat_valid.compute()
        self.confmat_valid.reset()
        # self.valid_loss.reset()

        acc, acc_mean = get_scores(confmat)
        # calculate valid_acc
        self.log("valid_acc_epoch", acc, prog_bar=True)
        self.log("valid_acc_mean_epoch", acc_mean, prog_bar=True)

    def test_step(self, batch, batch_idx):

        loss, count, confmat = self.forward(batch, training=False)

        self.confmat_test(confmat)  # confusion matrix of patches.
        self.log("test_loss", self.test_loss(loss, count), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
                 logger=False)

    def test_epoch_end(self, outputs):
        confmat = self.confmat_test.compute()
        # print(confmat)
        self.confmat_test.reset()
        # self.test_loss.reset()

        acc, acc_mean = get_scores(confmat, istest=True, save_conf=True, conf_name=os.path.join(os.getcwd(), "output_conf", self.ckpt_path)) # now only ops is supported
        # calculate valid_acc
        self.log("test_acc_epoch", acc, prog_bar=True)
        self.log("test_acc_mean_epoch", acc_mean, prog_bar=True)

    def configure_optimizers(self):
        # load optimiser based on mode
        self.optimizer = get_optimizer(self.net, self.mode)
        self.scheduler = PolyLRScheduler(self.optimizer, power=1.0, final_lr=0.0, warmup_steps=1500, num_epochs=self.trainer.max_epochs)
        return [self.optimizer]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items

    def on_save_checkpoint(self, checkpoint):
        checkpoint["optimizer"] = self.optimizer.state_dict()
        # checkpoint["sam_optimizer"] = self.optimizer.state_dict()
        # checkpoint["scheduler"] = self.scheduler.state_dict()
        checkpoint["state_dict"] = self.net.state_dict()
        checkpoint.pop("optimizer_states")
        checkpoint.pop("lr_schedulers")
