import pickle
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import AdsDataset
from preprocess.boxes import write_dict_to_json
from tools.evaluate import evaluate
from tools.engine import train_one_epoch
from tools.evaluate import evaluate
import tools.transforms as T
import tools.utils as utils


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


def create_train_test_dataset(dataset: AdsDataset, desc="sentiments"):
    """Split the dataset into training and testing

    Args:
        dataset (AdsDataset): a Pytorch Dataset

    Returns:
        (AdsDataset, AdsDataset): train dataset, test dataset
    """
    # randomly select the training and testing indices
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, train_size=0.85, shuffle=True, random_state=24)

    # split the dataset into train and test
    train_dataset = torch.utils.data.Subset(AdsDataset(descriptor=desc,
        transforms=get_transform(train=True)), train_indices)
    test_dataset = torch.utils.data.Subset(AdsDataset(descriptor=desc,
        transforms=get_transform(train=False)), test_indices)

    return train_dataset, test_dataset


def train(num_classes: int, num_epochs: int, checkpoint=None, batch_size=8, num_workers=1):
    """Train the model

    Args:
        num_classes (int): number of label classes
        num_epochs (int): number of epochs to train the model
        checkpoint (str, optional): path to the checkpoint file. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 8.
        num_workers (int, optional): number of workers. Defaults to 1.
    """
    # create dataset
    ads_dataset = AdsDataset()
    # create training & testing dataset
    train_dataset, test_dataset = create_train_test_dataset(ads_dataset)

    # define training data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=utils.collate_fn)

    # define testing data loaders
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        # load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # move model to the right device
    model.to(device)

    # construct a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    # create dicts to store statistics in json file
    metric_logs_avg, metric_logs_med = dict(), dict()

    # set print frequency
    print_freq = 10

    # training
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # train for one epoch, printing every 10 iterations
        print("Training epoch " + str(epoch) + " ...")
        curr_log = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=print_freq)

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
        evaluate(model, test_dataloader, device=device, print_freq=print_freq)

        print("Saving checkpoint")
        # save checkpoint
        utils.save_checkpoint(epoch, model, optimizer, filename='outputs/cp_fasterrcnn.pth.tar')
    write_dict_to_json("outputs/fasterrcnn_train_metric_log_med.json", metric_logs_med)
    write_dict_to_json("outputs/fasterrcnn_train_metric_log_avg.json", metric_logs_avg)


if __name__ == "__main__":
    le = pickle.loads(open("outputs/le.pickle", "rb").read())
    train(num_classes=len(le.classes_), num_epochs=6)