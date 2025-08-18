import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F



from .base import BaseEstimator



class Encoder(nn.Module):
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block='LSTM'):
        super(Encoder, self).__init__()
        if block == 'LSTM':
            self.model = nn.LSTM(number_of_features, hidden_size, hidden_layer_depth, dropout=dropout,
                                 batch_first=False)
        elif block == 'GRU':
            self.model = nn.GRU(number_of_features, hidden_size, hidden_layer_depth, dropout=dropout, batch_first=False)
        else:
            raise NotImplementedError

    def forward(self, x):
        _, (h_end, c_end) = self.model(x)
        # h_end is (num_layers * num_directions, batch, hidden_size)
        # We take the hidden state from the last layer
        return h_end[-1, :, :]


class Lambda(nn.Module):
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()
        self.hidden_to_mean = nn.Linear(hidden_size, latent_length)
        self.hidden_to_logvar = nn.Linear(hidden_size, latent_length)
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)
        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps * std + self.latent_mean
        else:
            # During inference, we return the mean
            return self.latent_mean


class Decoder(nn.Module):
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size,
                 block='LSTM', input_dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.output_size = output_size
        self.input_dropout_rate = input_dropout_rate

        self.model = nn.LSTM(output_size, hidden_size, hidden_layer_depth,
                             batch_first=False) if block == 'LSTM' else nn.GRU(output_size, hidden_size,
                                                                               hidden_layer_depth, batch_first=False)
        self.latent_to_hidden = nn.Linear(latent_length, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

        # c_0 is initialized on the fly in the forward pass
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, target_sequence):
        # The initial hidden state of the decoder is conditioned on the latent variable
        h_0 = torch.stack([self.latent_to_hidden(latent) for _ in range(self.hidden_layer_depth)])

        current_batch_size = latent.size(0)

        # Create a start-of-sequence (SOS) token (a zero vector)
        sos_token = torch.zeros(1, current_batch_size, self.output_size, device=latent.device, dtype=latent.dtype)

        # Shift target sequence for teacher forcing and prepend SOS token
        decoder_inputs = torch.cat([sos_token, target_sequence[:-1, :, :]], dim=0)

        # Apply input dropout (word dropout)
        if self.training:
            # Create a mask to zero out entire timesteps for certain batch elements
            dropout_mask = (
                        torch.rand(decoder_inputs.shape[:2], device=latent.device) > self.input_dropout_rate).unsqueeze(
                -1)
            decoder_inputs = decoder_inputs * dropout_mask

        if isinstance(self.model, nn.LSTM):
            # LSTM requires a cell state as well, which we initialize to zeros
            c_0_batch = torch.zeros(self.hidden_layer_depth, current_batch_size, self.hidden_size, device=latent.device)
            decoder_output, _ = self.model(decoder_inputs, (h_0, c_0_batch))
        else:  # GRU
            decoder_output, _ = self.model(decoder_inputs, h_0)

        return self.hidden_to_output(decoder_output)


class IWAE(BaseEstimator, nn.Module):
    """
    Importance-Weighted Autoencoder for Time Series Data.
    """

    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM', n_epochs=5, dropout_rate=0.,
                 optimizer='Adam', loss='MSELoss', cuda=False, print_every=100, clip=True,
                 max_grad_norm=5, dload='.', beta=1.0, kl_annealing_epochs=15, input_dropout_rate=0.25,
                 sigmoid_anneal_k=1.5, sigmoid_anneal_x0=0.9,
                 # New parameter for IWAE
                 k=5):
        super(IWAE, self).__init__()

        self.dtype = torch.cuda.FloatTensor if cuda and torch.cuda.is_available() else torch.FloatTensor
        self.use_cuda = cuda and torch.cuda.is_available()

        self.k = k  # Number of importance samples

        self.encoder = Encoder(number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout_rate, block)
        self.lmbd = Lambda(hidden_size, latent_length)
        self.decoder = Decoder(sequence_length, batch_size * k, hidden_size, hidden_layer_depth, latent_length,
                               number_of_features, block, input_dropout_rate)

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.latent_length = latent_length
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.kl_annealing_epochs = kl_annealing_epochs
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.dload = dload
        self.print_every = print_every
        self.train_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.valid_losses = []
        self._beta_final = beta
        self.current_beta = 0.0
        self.is_fitted = False

        # Sigmoid Annealing Parameters
        self.sigmoid_anneal_k = sigmoid_anneal_k
        self.sigmoid_anneal_x0 = sigmoid_anneal_x0

        if self.use_cuda:
            self.cuda()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) if optimizer == 'Adam' else optim.SGD(
            self.parameters(), lr=learning_rate)

        # We need per-element loss for IWAE calculation.
        if loss == 'MSELoss':
            self.recon_loss_fn = nn.MSELoss(reduction='none')
        elif loss == 'SmoothL1Loss':
            self.recon_loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError("Loss must be 'MSELoss' or 'SmoothL1Loss'")

    def forward(self, x):
        """
        Forward pass for inference. Uses the mean of the latent distribution.
        """
        self.eval()  # Ensure we are in eval mode for single-sample generation
        with torch.no_grad():
            latent = self.lmbd(self.encoder(x))
            x_decoded, _ = self.decoder(latent, x)
        return x_decoded, latent

    def _get_log_prob(self, dist, value):
        """
        Calculates the log probability of a value given a normal distribution.
        """
        # Assuming dist is a tuple of (mean, logvar)
        mean, logvar = dist
        # Expand dimensions for k samples
        mean = mean.unsqueeze(0)  # (1, batch, latent)
        logvar = logvar.unsqueeze(0)  # (1, batch, latent)

        # log N(value; mu, sigma^2)
        return -0.5 * (
                logvar + (value - mean).pow(2) / logvar.exp() + np.log(2 * np.pi)
        ).sum(dim=-1)  # Sum over latent dimension

    def compute_loss(self, x):
        """
        Computes the IWAE loss for a batch of data.
        This function contains the multi-sample forward pass used for training.
        """
        # Get latent distribution parameters from the encoder
        encoder_output = self.encoder(x)
        latent_mean = self.lmbd.hidden_to_mean(encoder_output)
        latent_logvar = self.lmbd.hidden_to_logvar(encoder_output)

        # Sample k times from the latent distribution
        std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn(self.k, self.batch_size, self.latent_length, device=x.device, dtype=self.dtype)
        z = latent_mean.unsqueeze(0) + eps * std.unsqueeze(0)  # Shape: (k, batch, latent)

        # Reshape for decoder
        # (k, batch, latent) -> (k * batch, latent)
        z_reshaped = z.view(self.k * self.batch_size, self.latent_length)
        # Repeat original input x for each of the k samples
        # (seq, batch, feat) -> (seq, k * batch, feat)
        x_repeated = x.repeat(1, self.k, 1)

        # Decode all k samples
        x_decoded = self.decoder(z_reshaped, x_repeated)

        # --- IWAE Loss Calculation ---
        # Reshape decoded output for loss calculation
        # (seq, k * batch, feat) -> (k, batch, seq, feat)
        x_decoded_reshaped = x_decoded.view(self.sequence_length, self.k, self.batch_size, -1).permute(1, 2, 0, 3)
        # Reshape original input for broadcasting
        # (seq, batch, feat) -> (1, batch, seq, feat)
        x_expanded = x.permute(1, 0, 2).unsqueeze(0)

        # 1. log p(x|z) - Reconstruction log-probability
        # recon_loss_fn returns (k, batch, seq, feat), sum over seq and feat dims
        log_p_x_given_z = -self.recon_loss_fn(x_decoded_reshaped, x_expanded).sum(dim=[-1, -2])

        # 2. log p(z) - Prior log-probability
        # Prior is a standard normal N(0, I)
        log_p_z = self._get_log_prob((torch.zeros_like(z), torch.zeros_like(z)), z)

        # 3. log q(z|x) - Posterior log-probability
        log_q_z_given_x = self._get_log_prob((latent_mean, latent_logvar), z)

        # --- KL divergence for logging ---
        # This is the standard VAE KL, not used in IWAE loss directly but useful for monitoring
        kl_div = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())

        # Combine for importance weight: log(w) = log p(x|z) + log p(z) - log q(z|x)
        # Apply beta annealing to the KL-like term
        log_weight = log_p_x_given_z + self.current_beta * (log_p_z - log_q_z_given_x)

        # IWAE objective: log (1/k * sum(w_i))
        # Use log-sum-exp for numerical stability
        log_evidence = torch.logsumexp(log_weight, dim=0) - torch.log(
            torch.tensor(self.k, device=x.device, dtype=self.dtype))

        # Final loss is the negative mean of the log-evidence estimates
        total_loss = -torch.mean(log_evidence)

        # For logging purposes, calculate average reconstruction loss
        recon_loss_for_logging = F.mse_loss(x_decoded, x_repeated)

        return total_loss, recon_loss_for_logging, kl_div

    def _train(self, train_loader, epoch):
        self.train()  # Set model to training mode
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        # Sigmoid Annealing for beta
        if epoch < self.kl_annealing_epochs:
            x_anneal = (epoch / self.kl_annealing_epochs)
            sigmoid_factor = 1 / (1 + np.exp(-self.sigmoid_anneal_k * (x_anneal - self.sigmoid_anneal_x0) * 10))
            self.current_beta = self._beta_final * sigmoid_factor
        else:
            self.current_beta = self._beta_final

        for t, (X_batch,) in enumerate(train_loader):
            # VRAE expects (seq_len, batch, features)
            X_batch = X_batch.permute(1, 0, 2).type(self.dtype)

            if self.use_cuda:
                X_batch = X_batch.cuda()

            self.optimizer.zero_grad()

            # Compute IWAE loss
            loss, recon_loss, kl_loss = self.compute_loss(X_batch)

            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            if (t + 1) % self.print_every == 0:
                print(
                    f'Batch {t + 1}, loss={loss.item():.4f}, recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}, beta={self.current_beta:.4f}')

        n_batches = t + 1
        return epoch_loss / n_batches, epoch_recon_loss / n_batches, epoch_kl_loss / n_batches

    # The rest of the methods (_validate, fit, _batch_transform, etc.) can remain
    # the same as in your VRAE class, as they correctly use the model in eval mode.
    # They are included here for completeness.

    def _validate(self, valid_loader):
        self.eval()  # Set model to evaluation mode
        epoch_loss = 0
        with torch.no_grad():
            for t, (X_batch,) in enumerate(valid_loader):
                X_batch = X_batch.permute(1, 0, 2).type(self.dtype)
                if self.use_cuda:
                    X_batch = X_batch.cuda()

                # Use the same IWAE loss for validation to have a consistent metric
                loss, _, _ = self.compute_loss(X_batch)
                epoch_loss += loss.item()
        return epoch_loss / (t + 1)

    def fit(self, train_dataset, valid_dataset=None, save=False):
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        if valid_dataset:
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")

            train_loss, train_recon, train_kl = self._train(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_recon_losses.append(train_recon)
            self.train_kl_losses.append(train_kl)

            epoch_summary = f"  → Avg Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Beta: {self.current_beta:.4f})"

            if valid_dataset:
                valid_loss = self._validate(valid_loader)
                self.valid_losses.append(valid_loss)
                epoch_summary += f" | Avg Valid Loss: {valid_loss:.4f}"

            print(epoch_summary)

        self.is_fitted = True
        if save:
            self.save(os.path.join(self.dload, 'model.pth'))

    def _batch_transform(self, x):
        """
        Transforms a single batch of data into the latent space.
        Uses the mean of the latent distribution (eval mode).
        """
        self.eval()
        with torch.no_grad():
            x = x.type(self.dtype)
            if self.use_cuda:
                x = x.cuda()
            # The lambda layer in eval mode returns the mean
            return self.lmbd(self.encoder(x)).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Reconstructs a single batch of data.
        Uses the mean of the latent distribution (eval mode).
        """
        self.eval()
        with torch.no_grad():
            x = x.type(self.dtype)
            if self.use_cuda:
                x = x.cuda()
            # The forward pass in eval mode uses the latent mean
            x_decoded, _ = self(x)
            return x_decoded.cpu().data.numpy()

    def transform(self, dataset, save=False):
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before calling transform.")
        z_run = [self._batch_transform(x[0].permute(1, 0, 2)) for x in loader]
        z_run = np.concatenate(z_run, axis=0)
        if save:
            np.save(os.path.join(self.dload, "z_run.npy"), z_run)
        return z_run

    def reconstruct(self, dataset, save=False):
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before calling reconstruct.")
        x_decoded_list = [self._batch_reconstruct(x[0].permute(1, 0, 2)) for x in loader]
        # Re-permute back to (batch, seq_len, features) and concatenate
        x_decoded = np.concatenate([x.transpose(1, 0, 2) for x in x_decoded_list], axis=0)
        if save:
            np.save(os.path.join(self.dload, "x_decoded.npy"), x_decoded)
        return x_decoded

    def fit_transform(self, dataset, save=False):
        self.fit(dataset, save=save)
        return self.transform(dataset, save=save)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
