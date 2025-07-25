import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from .base import BaseEstimator

class Encoder(nn.Module):
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block='LSTM'):
        super(Encoder, self).__init__()
        if block == 'LSTM':
            self.model = nn.LSTM(number_of_features, hidden_size, hidden_layer_depth, dropout=dropout)
        elif block == 'GRU':
            self.model = nn.GRU(number_of_features, hidden_size, hidden_layer_depth, dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        _, (h_end, _) = self.model(x)
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
            return self.latent_mean


class Decoder(nn.Module):
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype,
                 block='LSTM', input_dropout_rate = 0.1):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_layer_depth = hidden_layer_depth
        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.output_size = output_size
        self.input_dropout_rate = input_dropout_rate

        self.model = nn.LSTM(output_size, hidden_size, hidden_layer_depth) if block == 'LSTM' else nn.GRU(output_size, hidden_size,
                                                                                                hidden_layer_depth)
        self.latent_to_hidden = nn.Linear(latent_length, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

        #self.register_buffer('decoder_inputs', torch.zeros(sequence_length, batch_size, 1))
        self.register_buffer('c_0', torch.zeros(hidden_layer_depth, batch_size, hidden_size))

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, target_sequence):
        h_0 = torch.stack([self.latent_to_hidden(latent) for _ in range(self.hidden_layer_depth)])

        current_batch_size = latent.size(0)

        # Create a start-of-sequence (SOS) token (a zero vector)
        sos_token = torch.zeros(1, current_batch_size, self.output_size, device=latent.device, dtype=latent.dtype)

        # Shift target sequence for teacher forcing and prepend SOS token
        decoder_inputs = torch.cat([sos_token, target_sequence[:-1, :, :]], dim=0)

        # Apply input dropout
        if self.training:
            # Create a mask to zero out entire timesteps
            dropout_mask = (
                        torch.rand(decoder_inputs.shape[:2], device=latent.device) > self.input_dropout_rate).unsqueeze(
                -1)
            decoder_inputs = decoder_inputs * dropout_mask

        if isinstance(self.model, nn.LSTM):
            c_0_batch = self.c_0[:, :current_batch_size, :]
            decoder_output, _ = self.model(decoder_inputs, (h_0, c_0_batch))
        else:
            decoder_output, _ = self.model(decoder_inputs, h_0)

        return self.hidden_to_output(decoder_output)


class VRAE(BaseEstimator, nn.Module):
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM', n_epochs=5, dropout_rate=0.,
                 optimizer='Adam', loss='MSELoss', cuda=False, print_every=100, clip=True,
                 max_grad_norm=5, dload='.', beta=1.0, kl_annealing_epochs=15, input_dropout_rate=0.25,
                 # New parameter for sigmoid annealing steepness
                 sigmoid_anneal_k=1.5, sigmoid_anneal_x0=0.9):
        super(VRAE, self).__init__()

        self.dtype = torch.cuda.FloatTensor if cuda and torch.cuda.is_available() else torch.FloatTensor
        self.use_cuda = cuda and torch.cuda.is_available()
        self.encoder = Encoder(number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout_rate, block)
        self.lmbd = Lambda(hidden_size, latent_length)
        self.decoder = Decoder(sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length,
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
        self.sigmoid_anneal_k = sigmoid_anneal_k  # Steepness of the sigmoid curve
        self.sigmoid_anneal_x0 = sigmoid_anneal_x0  # Midpoint of the sigmoid curve (fraction of kl_annealing_epochs)

        if self.use_cuda:
            self.cuda()
            #self.decoder.decoder_inputs = self.decoder.decoder_inputs.cuda()
            self.decoder.c_0 = self.decoder.c_0.cuda()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) if optimizer == 'Adam' else optim.SGD(
            self.parameters(), lr=learning_rate)

        if loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError("Loss must be 'MSELoss' or 'SmoothL1Loss'")
        # --- END MODIFICATION ---

    def forward(self, x):
        latent = self.lmbd(self.encoder(x))
        return self.decoder(latent,x), latent

    def _rec(self, x_decoded, x):
        mean, logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        recon_loss = self.loss_fn(x_decoded, x)
        total_loss = recon_loss + self.current_beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def compute_loss(self, x):
        x = x.type(self.dtype)
        x_decoded, _ = self(x)
        return self._rec(x_decoded, x)

    def _train(self, train_loader, epoch):
        self.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        # --- Sigmoid Annealing Logic ---
        if epoch < self.kl_annealing_epochs:
            # Normalized epoch progress from 0 to 1
            x = (epoch / self.kl_annealing_epochs)
            # Apply sigmoid function: 1 / (1 + exp(-k * (x - x0)))
            # k controls steepness, x0 controls midpoint
            sigmoid_factor = 1 / (1 + np.exp(-self.sigmoid_anneal_k * (
                        x - self.sigmoid_anneal_x0) * 10))  # Multiplied by 10 for a typical sigmoid range
            self.current_beta = self._beta_final * sigmoid_factor
        else:
            self.current_beta = self._beta_final
        # --- End Sigmoid Annealing Logic ---

        for t, X in enumerate(train_loader):
            X = X[0].permute(1, 0, 2)

            if self.use_cuda:
                X = X.cuda()

            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss = self.compute_loss(X)
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

    def _validate(self, valid_loader):
        self.eval()
        epoch_loss = 0
        with torch.no_grad():
            for t, X in enumerate(valid_loader):
                X = X[0].permute(1, 0, 2)
                if self.use_cuda:
                    X = X.cuda()
                loss, _, _ = self.compute_loss(X)
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

            epoch_summary = f"  → Avg Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Current Beta: {self.current_beta:.4f})"

            if valid_dataset:
                valid_loss = self._validate(valid_loader)
                self.valid_losses.append(valid_loss)
                epoch_summary += f" | Avg Valid Loss: {valid_loss:.4f}"

            print(epoch_summary)

        self.is_fitted = True
        if save:
            self.save(os.path.join(self.dload, 'model.pth'))

    def _batch_transform(self, x):
        self.eval()
        with torch.no_grad():
            x = x.type(self.dtype)
            if self.use_cuda:
                x = x.cuda()
            return self.lmbd(self.encoder(x)).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            x = x.type(self.dtype)
            if self.use_cuda:
                x = x.cuda()
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
        x_decoded = [self._batch_reconstruct(x[0].permute(1, 0, 2)) for x in loader]
        x_decoded = np.concatenate(x_decoded,
                                   axis=1)
        if save:
            np.save(os.path.join(self.dload, "x_decoded.npy"), x_decoded)
        return x_decoded

    def fit_transform(self, dataset, save=False):
        self.fit(dataset, save=save)
        return self.transform(dataset, save=save)