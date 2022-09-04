import os
import losses
import feedforward
import trackers_info_dataset
import pose_dataset
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Hyperparameters
use_tune = False
use_adam = False
epochs = 5
filename_input = "data/direction_short_jl.onnx"
loss_type = "mse"  # "mse" or "dot"
gamma = 0.95  # Decay factor for the learning rate
# Learning
config = {
    "batch_size": tune.choice([64]),
    "hidden_size": tune.choice([32, 64, 128]),
    "number_hidden_layers": tune.choice([2]),
    "learning_rate": tune.loguniform(1e-4, 1e-3),
    "weight_decay": tune.loguniform(1e-2, 1),
    # "momentum": tune.uniform(0.0, 0.99),
}
default_config = {
    "batch_size": 64,
    "hidden_size": 32,
    "number_hidden_layers": 2,
    "learning_rate": 0.0003,
    "weight_decay": 0.035,
    "momentum": 0.9,
}
# Recursive Learning
number_recursions = 10
assert number_recursions >= 1

path_training = "PATH_TO_YOUR_PROJECT/MMVR/Assets/MMData/Data/TrainingMSData/"
path_test = (
    "PATH_TO_YOUR_PROJECT/MMVR/Assets/MMData/Data/TestMSData/"
)


# Dataloader
class dataset_input(Dataset):
    def __init__(self, trackers_info):
        input = np.arange(
            1, trackers_info.shape[0] - number_recursions, 1, dtype=np.longlong
        )
        self.input = torch.from_numpy(input)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx]


# Training

# Hyperparameter tuning
if use_tune:
    scheduler_tuning = ASHAScheduler(
        metric="loss", mode="min", max_t=epochs, grace_period=5, reduction_factor=2
    )
    reporter = CLIReporter(metric_columns=["loss", "training_iteration"])


def train_direction(config):
    # Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Import Data
    trackers_input = trackers_info_dataset.trackers_info_dataset(
        path_training + "TrainingMSData.mstrackers"
    )
    trackers = trackers_input.info
    pose_dataset_input = pose_dataset.pose_dataset(
        path_training + "TrainingMSData.mspose"
    )
    poses = pose_dataset_input.poses
    poses_mean = pose_dataset_input.mean
    poses_std = pose_dataset_input.std

    trackers_test_input = trackers_info_dataset.trackers_info_dataset(
        path_test + "TestMSData.mstrackers"
    )
    trackers_test = trackers_test_input.info
    pose_test_dataset_input = pose_dataset.pose_dataset(path_test + "TestMSData.mspose")
    poses_test = pose_test_dataset_input.poses

    training_trackers = trackers
    training_poses = poses
    test_trackers = trackers_test
    test_poses = poses_test

    training_dataset = dataset_input(training_trackers)
    test_dataset = dataset_input(
        test_trackers
    )  # training_trackers_locomotion is the same because the input will be zeroed differently

    input_pose_size = training_trackers.shape[1] + 6
    output_pose_size = 6

    # Data
    train_dataloader = DataLoader(
        training_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True
    )

    # Model
    direction_model = feedforward.FeedForward(
        training_trackers,
        training_poses,
        test_trackers,
        test_poses,
        input_pose_size,
        config["hidden_size"],
        config["number_hidden_layers"],
        output_pose_size,
        number_recursions,
        device,
    ).to(device)

    # Loss
    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type == "dot":
        loss_fn = losses.dot_loss(poses_mean[:6], poses_std[:6], device)

    # Optimizer
    if use_adam:
        optimizer = torch.optim.AdamW(
            direction_model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            direction_model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

    # Checkpoint
    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(
    #         os.path.join(checkpoint_dir, "checkpoint")
    #     )
    #     direction_model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Training
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch) + " ----------------------------")
        direction_model.train()
        avg_train_loss = direction_model.train_loop(
            train_dataloader, loss_fn, optimizer
        )
        direction_model.eval()
        avg_test_loss = direction_model.test_loop(test_dataloader, loss_fn)
        if scheduler != None:
            scheduler.step()
        if use_tune:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((direction_model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=avg_test_loss)

    print("Finished Training")
    return direction_model


# Device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

# Import Data
trackers_input = trackers_info_dataset.trackers_info_dataset(
    path_training + "TrainingMSData.mstrackers"
)
trackers = trackers_input.info
pose_dataset_input = pose_dataset.pose_dataset(path_training + "TrainingMSData.mspose")
poses = pose_dataset_input.poses
poses_mean = pose_dataset_input.mean
poses_std = pose_dataset_input.std

trackers_test_input = trackers_info_dataset.trackers_info_dataset(
    path_test + "TestMSData.mstrackers"
)
trackers_test = trackers_test_input.info
pose_test_dataset_input = pose_dataset.pose_dataset(path_test + "TestMSData.mspose")
poses_test = pose_test_dataset_input.poses

training_trackers = trackers
training_poses = poses
test_trackers = trackers_test
test_poses = poses_test

training_dataset = dataset_input(training_trackers)
test_dataset = dataset_input(
    test_trackers
)  # training_trackers_locomotion is the same because the input will be zeroed differently

input_pose_size = training_trackers.shape[1] + 6
output_pose_size = 6

if use_tune:
    result = tune.run(
        # partial(
        #     train_direction,
        #     checkpoint_dir=checkpoint_dir,
        # ),
        train_direction,
        resources_per_trial={
            "cpu": 2,
            "gpu": 0.2,
        },  # if gpu < 1: Share GPU among trials (make sure there is enough memory)
        config=config,
        num_samples=10,
        scheduler=scheduler_tuning,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

    best_direction_model = feedforward.FeedForward(
        training_trackers,
        training_poses,
        test_trackers,
        test_poses,
        input_pose_size,
        best_trial.config["hidden_size"],
        best_trial.config["number_hidden_layers"],
        output_pose_size,
        number_recursions,
        device,
    ).to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, "checkpoint")
    )
    best_direction_model.load_state_dict(model_state)

    def test_best_model(direction_model, trial):
        test_dataloader = DataLoader(
            test_dataset, batch_size=trial.config["batch_size"], shuffle=True
        )
        if loss_type == "mse":
            loss_fn = nn.MSELoss()
        elif loss_type == "dot":
            loss_fn = losses.dot_loss(poses_mean[:6], poses_std[:6], device)
        test_losses = direction_model.test_loop(test_dataloader, loss_fn)
        return test_losses

    test_error = test_best_model(best_direction_model, best_trial)
    print("Best trial test set loss: {}".format(test_error))
else:
    best_direction_model = train_direction(default_config)

best_direction_model.save(input_pose_size, device, filename_input)
