import torch
import rotations_torch as rot


class dot_loss:
    def __init__(self, mean, std, device):
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device

    def __call__(self, predicted_dir, target_dir):
        # predicted_dir and target_dir are continuous (2-axis) rotations

        # Denormalize
        predicted_dir = predicted_dir * self.std + self.mean
        target_dir = target_dir * self.std + self.mean

        # Compute vectors
        predicted_rot = rot.continuous_to_mat(predicted_dir)
        target_rot = rot.continuous_to_mat(target_dir)
        forwards = torch.tensor([0.0, 0.0, 1.0]).to(self.device)
        forwards = forwards.repeat(predicted_rot.shape[0], 1)
        predicted_forward = rot.mul_mat_vec(predicted_rot, forwards)
        target_forward = rot.mul_mat_vec(target_rot, forwards)

        # Compute dot product
        return torch.mean(
            ((-(predicted_forward * target_forward).sum(-1))
            + 1  # +1 so it goes from 0 to 2
            ) / 2.0 # /2 so it goes from 0 to 1
        ) # the result is negated because we want to minimize the loss
