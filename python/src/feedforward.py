import torch
from torch import nn
import numpy as np


class FeedForward(nn.Module):
    def __init__(
        self,
        training_trackers,
        training_poses,
        test_trackers,
        test_poses,
        input_size,
        hidden_size,
        number_hidden_layers,
        output_size,
        number_recursions,
        device,
    ):
        super(FeedForward, self).__init__()

        self.training_trackers = torch.from_numpy(
            training_trackers.astype(np.float32)
        ).to(device)
        self.training_poses = torch.from_numpy(training_poses.astype(np.float32)).to(
            device
        )
        self.test_trackers = torch.from_numpy(test_trackers.astype(np.float32)).to(
            device
        )
        self.test_poses = torch.from_numpy(test_poses.astype(np.float32)).to(device)

        self.number_recursions = number_recursions
        self.input_size = input_size
        self.device = device
        self.number_hidden_layers = number_hidden_layers

        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        for i in range(self.number_hidden_layers):
            self.linear_stack.add_module(
                "linear" + str(i), nn.Linear(hidden_size, hidden_size)
            )
        self.linear_stack.add_module("linear_out", nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.linear_stack(x)

    def train_loop(self, train_dataloader, loss_fn, optimizer):
        size = len(train_dataloader.dataset)
        train_loss = 0

        for batch, (idx) in enumerate(train_dataloader):
            self.zero_grad()

            idx = idx.to(self.device)

            # Compute prediction
            input = torch.cat(
                (self.training_trackers[idx, :], self.training_poses[idx - 1, :6]),
                dim=-1,
            )
            for i in range(self.number_recursions):
                predicted_dir = self(input)
                input = self.training_trackers[idx + (i + 1), :]
                input = torch.cat((input, predicted_dir), dim=-1)

            loss = loss_fn(
                predicted_dir,
                self.training_poses[idx + self.number_recursions - 1, :6],
            )
            train_loss += loss.item()  # mean of losses in this batch

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Print progress
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(idx)
                print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        return train_loss / len(train_dataloader)  # divide by number of batches

    def test_loop(self, test_dataloader, loss_fn):
        num_batches = len(test_dataloader)
        test_loss = 0

        with torch.no_grad():
            for idx in test_dataloader:
                idx = idx.to(self.device)

                # Compute prediction
                input = torch.cat(
                    (
                        self.test_trackers[idx, :],
                        self.test_poses[idx - 1, :6],
                    ),
                    dim=-1,
                )
                for i in range(self.number_recursions):
                    predicted_dir = self(input)
                    input = self.test_trackers[idx + (i + 1), :]
                    input = torch.cat((input, predicted_dir), dim=-1)

                loss = loss_fn(
                    predicted_dir,
                    self.test_poses[idx + self.number_recursions - 1, :6],
                )
                test_loss += loss.item()

        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f}")
        return test_loss

    def save(self, input_size, device, path_filename):
        # Save model
        torch.onnx.export(
            self,
            torch.randn(1, input_size, device=device),  # dummy input
            path_filename,
            export_params=True,
            opset_version=9,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["direction"],
        )
