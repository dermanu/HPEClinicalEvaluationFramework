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
    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

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

    d = 1 - traceTA**2
    c = muX - b * np.dot(muY, T)

    Z = b * np.dot(Y0, T) + muX
    return d, Z

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

# Dummy example
num_joints = 28 # total number of joints (dataset1 + dataset 2)
num_latent = np.ceil(np.sqrt(num_joints)*2)
acae = ACAE(num_joints, num_latent)
optimizer = torch.optim.Adam(acae.parameters(), lr=0.001)
dummy_data_loader = [torch.randn(10, num_joints, 3) for _ in range(100)]  # Example data
train_acae(acae, dummy_data_loader, optimizer)
