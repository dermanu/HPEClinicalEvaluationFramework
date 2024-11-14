import torch
import torch.nn as nn
import numpy as np


class ACAE(nn.Module):
    def __init__(self, num_joints, num_latent):
        super(ACAE, self).__init__()
        self.num_joints = num_joints
        self.num_latent = num_latent

        # Initialize parameters with Xavier initialization for better convergence
        self.Wenc = nn.Parameter(torch.empty(num_latent, num_joints))
        nn.init.xavier_uniform_(self.Wenc)
        self.Wdec = nn.Parameter(torch.empty(num_joints, num_latent))
        nn.init.xavier_uniform_(self.Wdec)

        print(f"Wenc shape: {self.Wenc.shape}")
        print(f"Wdec shape: {self.Wdec.shape}")

    def forward(self, x):
        batch_size = x.size(0)

        # Normalize weights to ensure they sum to one and are used equally across coordinates
        Wenc_normalized = torch.softmax(self.Wenc, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        Wdec_normalized = torch.softmax(self.Wdec, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)

        # Ensure x has shape [batch_size, num_joints, 3]
        print(f"Input shape: {x.shape}")
        x = x.permute(0, 2, 1)  # Shape: [batch_size, 3, num_joints]
        print(f"Permuted input shape: {x.shape}")

        # Print weight shapes
        print(f"Wenc_normalized shape: {Wenc_normalized.shape}")
        print(f"Wdec_normalized shape: {Wdec_normalized.shape}")

        # Perform batch matrix multiplication
        latent = torch.bmm(Wenc_normalized, x)  # Shape: [batch_size, num_latent, 3]
        print(f"Latent shape after bmm with Wenc_normalized: {latent.shape}")
        latent = latent.permute(0, 2, 1)  # Shape: [batch_size, 3, num_latent]
        print(f"Latent shape after permute: {latent.shape}")

        reconstruction = torch.bmm(Wdec_normalized, latent)  # Shape: [batch_size, num_joints, 3]
        print(f"Reconstruction shape after bmm with Wdec_normalized: {reconstruction.shape}")
        reconstruction = reconstruction.permute(0, 2, 1)  # Shape: [batch_size, num_joints, 3]
        print(f"Reconstruction shape after permute: {reconstruction.shape}")

        return reconstruction, latent

    def reconstruction_loss(self, original, reconstructed):
        return torch.mean(torch.abs(original - reconstructed))

    def regularization_loss(self):
        return torch.abs(torch.norm(self.Wenc, p=1) + torch.norm(self.Wdec, p=1))


def procrustes(X, Y, scaling=True):
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        b = traceTA * normX / normY
    else:
        b = 1

    d = 1 - traceTA ** 2

    Z = b * np.dot(Y0, T) + muX
    return d, Z


def normalize_and_align(data):
    try:
        base = data[0].numpy()
        aligned_data = []
        for x in data:
            _, Z = procrustes(base, x.numpy(), scaling=True)
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


# Dummy example
num_joints = 28  # total number of joints (dataset1 + dataset 2)
num_latent = int(np.ceil(np.sqrt(num_joints) * 2))
acae = ACAE(num_joints, num_latent)
optimizer = torch.optim.Adam(acae.parameters(), lr=0.001)
dummy_data_loader = [torch.randn(32, num_joints, 3) for _ in range(100)]  # Example data
train_acae(acae, dummy_data_loader, optimizer)
