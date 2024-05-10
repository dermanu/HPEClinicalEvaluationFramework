import torch
import torch.nn as nn
import numpy as np

class ACAE(nn.Module):
    def __init__(self, num_joints, num_latent, symmetric_pairs=None):
        super(ACAE, self).__init__()
        self.num_joints = num_joints
        self.num_latent = num_latent
        self.symmetric_pairs = symmetric_pairs

        # Initialize parameters with Xavier initialization for better convergence
        self.Wenc = nn.Parameter(torch.empty(num_latent, num_joints))
        nn.init.xavier_uniform_(self.Wenc)
        self.Wdec = nn.Parameter(torch.empty(num_joints, num_latent))
        nn.init.xavier_uniform_(self.Wdec)

    def forward(self, x):
        self.Wenc_normalized = torch.softmax(self.Wenc, dim=1).unsqueeze(2).repeat(1, 1, 3)
        self.Wdec_normalized = torch.softmax(self.Wdec, dim=1).unsqueeze(2).repeat(1, 1, 3)
        latent = torch.bmm(self.Wenc_normalized, x)
        reconstruction = torch.bmm(self.Wdec_normalized, latent)
        return reconstruction, latent

    def reconstruction_loss(self, original, reconstructed):
        return torch.mean(torch.abs(original - reconstructed))

    def regularization_loss(self):
        return torch.norm(self.Wenc, p=1) + torch.norm(self.Wdec, p=1)

def procrustes(X, Y, scaling=True, reflection='best'):
    # Similar to before but with potential for handling batches if necessary
    # (omitting for brevity)

def normalize_and_align(data):
    # Updated error handling
    try:
        reference_shape = data[0]
        aligned_data = []
        for x in data:
            _, Z = procrustes(reference_shape, x.numpy(), scaling=True)
            aligned_data.append(torch.tensor(Z, dtype=torch.float32))
        return torch.stack(aligned_data)
    except Exception as e:
        print(f"Error in normalize_and_align: {e}")
        return data  # Fallback to original data

def train_acae(acae, data_loader, optimizer, epochs=10, lambda_sparse=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            if not isinstance(data, torch.Tensor):
                print("Skipping invalid data batch")
                continue

            optimizer.zero_grad()
            data_aligned = normalize_and_align(data)
            reconstructed, latent = acae(data_aligned)
            recon_loss = acae.reconstruction_loss(data_aligned, reconstructed)
            reg_loss = acae.regularization_loss()
            loss = recon_loss + lambda_sparse * reg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')

# Example usage remains similar
