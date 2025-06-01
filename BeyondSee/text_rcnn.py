import torch
from torch import nn
from torch.jit.annotations import Tuple, List
from tfrcnn_roi_heads import TFRCNNRoIHeads, TFRCNNBoxHead


class TextFasterRCNN(nn.Module):
    """
    Ensemble model to combine Faster RCNN with text descriptors.

    """

    def __init__(self, faster_rcnn, text_embed_size=0):
        super(TextFasterRCNN, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.transform = faster_rcnn.transform
        self.text_embed_size = text_embed_size

        # Box Head
        out_channels = faster_rcnn.backbone.out_channels
        resolution = faster_rcnn.roi_heads.box_roi_pool.output_size[0]
        representation_size = 1024
        self.box_head = TFRCNNBoxHead(
            out_channels * resolution ** 2 + self.text_embed_size,
            representation_size
        )

        # ROI Heads
        self.roi_heads = TFRCNNRoIHeads(
            # Box
            faster_rcnn.roi_heads.box_roi_pool,
            self.box_head,
            faster_rcnn.roi_heads.box_predictor,
            # box_fg_iou_thresh
            faster_rcnn.roi_heads.proposal_matcher.high_threshold,
            # box_bg_iou_thresh
            faster_rcnn.roi_heads.proposal_matcher.low_threshold,
            # box_batch_size_per_image,
            faster_rcnn.roi_heads.fg_bg_sampler.batch_size_per_image,
            # box_positive_fraction,
            faster_rcnn.roi_heads.fg_bg_sampler.positive_fraction,
            # bbox_reg_weights,
            faster_rcnn.roi_heads.box_coder.weights,
            # box_score_thresh,
            faster_rcnn.roi_heads.score_thresh,
            # box_nms_thresh,
            faster_rcnn.roi_heads.nms_thresh,
            # box_detections_per_img,
            faster_rcnn.roi_heads.detections_per_img
        )

    def forward(self, images, descriptors, targets=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # Register forward hook for:
        #   - features (self.backbone(images.tensors))
        #   - proposals (self.rpn(images, features, targets))
        #   - box features (box roi pool output)

        activation = {}

        def get_activation(name):
            # the hook signature
            def hook(model, input, output):
                activation[name] = output
            return hook

        h1 = self.faster_rcnn.backbone\
            .register_forward_hook(get_activation('backbone'))
        h2 = self.faster_rcnn.rpn.register_forward_hook(get_activation('rpn'))
        h3 = self.faster_rcnn.roi_heads.box_roi_pool\
            .register_forward_hook(get_activation('box_roi_pool'))

        # Set Faster RCNN model to evaluation mode as we do not want to update
        # its weights
        self.faster_rcnn.eval()
        self.faster_rcnn(images)
        #self.faster_rcnn(images.to("cuda"))

        features = activation['backbone']
        proposals, proposal_losses = activation['rpn']
        box_features = activation['box_roi_pool']

        images, targets = self.transform(images, targets)

        # Transform descriptors into a tensor.
        descriptors = torch.stack(descriptors, dim=0).to("cuda")
        #descriptors = torch.stack(descriptors, dim=0)

        detections, detector_losses = self.roi_heads(features, proposals,
                                                     images.image_sizes,
                                                     descriptors,
                                                     targets)
        detections = self.transform.postprocess(detections, images.image_sizes,
                                                original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # detach the hooks
        h1.remove()
        h2.remove()
        h3.remove()

        if self.training:
            return losses

        return detections

