import pickle
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import AdsDataset
from preprocess.boxes import write_dict_to_json
from preprocess.descriptors import SentimentPreProcessor, load_annotation_json
from sklearn.model_selection import train_test_split
import tools.transforms as T
import tools.utils as utils
from text_rcnn import TextFasterRCNN
from tools.engine import train_one_epoch
from tools.evaluate import evaluate
import math
from torch.multiprocessing import set_start_method


def get_transform(train: bool):
    """Return the transform function

    Args:
        train (bool): whether the transform is applied on training dataset

    Returns:
        func: transform function on image and target
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def create_train_test_dataset(dataset: AdsDataset, descriptor="sentiments"):
    """Split the dataset into training and testing

    Args:
        dataset (AdsDataset): a Pytorch Dataset
        descriptor (descriptor): the descriptor
    Returns:
        (AdsDataset, AdsDataset): train dataset, test dataset
    """
    # randomly select the training and testing indices
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, train_size=0.85, shuffle=True, random_state=24)

    # split the dataset into train and test
    train_dataset = torch.utils.data.Subset(AdsDataset(
        transforms=get_transform(train=True), descriptor=descriptor),
        train_indices)
    test_dataset = torch.utils.data.Subset(AdsDataset(
        transforms=get_transform(train=False), descriptor=descriptor),
        test_indices)

    return train_dataset, test_dataset


def train(num_classes: int, num_epochs: int,
          faster_rcnn_trained=None, checkpoint=None,
          descriptor="sentiments",
          batch_size=8, num_workers=1):
    """
    Train the model.

    :param num_classes:         (int) number of label classes
    :param num_epochs:          (int) number of epochs to train
    :param faster_rcnn_trained: trained faster r-cnn model
    :param checkpoint:          (str, optional) path to the checkpoint file.
    :param descriptor:          (str) descriptor to add to ensemble model
    :param batch_size:          (int, optional) batch size.
    :param num_workers:         (int, optional) number of workers.
    """

    # Create the dataset
    ads_dataset = AdsDataset(descriptor=descriptor)
    # Get the text embedding size
    text_embed_size = ads_dataset.descriptor_preprocessor.embed_size
    # Create training and testing dataset
    train_dataset, test_dataset = create_train_test_dataset(dataset=ads_dataset,
                                                            descriptor=descriptor)

    # Define training data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=utils.collate_fn)

    # Define testing data loaders
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    # In case GPU is provided
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialise model or load checkpoint
    if checkpoint is None:
        # Check for the trained Faster RCNN model.
        if faster_rcnn_trained is None:
            # get the model
            faster_rcnn_trained = torchvision.models.detection\
                .fasterrcnn_resnet50_fpn(pretrained=True)
            # get the number of input features for the classifier
            in_features = faster_rcnn_trained.roi_heads.box_predictor\
                .cls_score.in_features
            # replace the pre-trained head with a new one
            faster_rcnn_trained.roi_heads.box_predictor = \
                FastRCNNPredictor(in_features, num_classes)
        else:
            faster_rcnn_checkpoint = torch.load(faster_rcnn_trained)
            # Get the model
            faster_rcnn_trained = faster_rcnn_checkpoint['model']
            faster_rcnn_trained = faster_rcnn_trained.to(device)

        start_epoch = 0
        # Create the ensemble model
        model = TextFasterRCNN(faster_rcnn=faster_rcnn_trained,
                               text_embed_size=text_embed_size)

        # Construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move model to the right device
    model.to(device)

    # Construct a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)
    lf = lambda x: (((1 + math.cos(x * math.pi / num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_weights)//batch_size)

    # Create dicts to store statistics in json file
    metric_logs_avg, metric_logs_med = dict(), dict()

    # Set print frequency
    print_freq = 10

    # Training
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # train for one epoch, printing every 10 iterations
        print("Training epoch " + str(epoch) + " ...")
        curr_log = train_one_epoch(model, optimizer, train_dataloader, device,
                                   epoch, print_freq=print_freq,
                                   include_descriptors=True)

        log_meters = curr_log.meters
        # nested dicts for current epoch statistics
        curr_median, curr_avg = dict(), dict()

        for key, value in log_meters.items():
            curr_median[key] = str(value.median)
            curr_avg[key] = str(value.global_avg)

        # Add nested dict to final dict
        metric_logs_med[str(epoch)] = curr_median
        metric_logs_avg[str(epoch)] = curr_avg

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, test_dataloader, device=device, print_freq=print_freq,
                 include_descriptors=True)

        print("Saving checkpoint")
        # save checkpoint
        checkpoint_name = "outputs/cp_{}_tfasterrcnn.pth.tar"\
            .format(descriptor)
        utils.save_checkpoint(epoch, model, optimizer,
                              filename=checkpoint_name)

    metric_logs_med_name = "outputs/{}_train_metric_log_med.json"\
        .format(descriptor)
    metric_logs_avg_name = "outputs/{}_train_metric_log_avg.json"\
        .format(descriptor)
    write_dict_to_json(metric_logs_med_name, metric_logs_med)
    write_dict_to_json(metric_logs_avg_name, metric_logs_avg)


if __name__ == "__main__":
    le = pickle.loads(open("outputs/le.pickle", "rb").read())
    set_start_method('spawn', force=True)
    train(num_classes=len(le.classes_),
          faster_rcnn_trained="outputs/cp_fasterrcnn_6ep.pth.tar",
          num_epochs=3)
