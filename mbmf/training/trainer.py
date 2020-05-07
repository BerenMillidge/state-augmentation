import os
import torch


class Trainer(object):
    def __init__(
        self,
        model,
        buffer,
        n_train_epochs=100,
        batch_size=50,
        learning_rate=1e-3,
        epsilon=1e-8,
        grad_clip_norm=1000,
    ):
        self.model = model
        self.buffer = buffer
        self.n_train_epochs = n_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm

        self.params = list(model.parameters())
        self.optim = torch.optim.Adam(self.params, lr=learning_rate, eps=epsilon)

    def save_models(self, epsiode):
        """ @TODO """
        pass

    def set_buffer(self, buffer):
        self.buffer = buffer

    def save_model(self, logdir):
        path = logdir + "buffer.pth"
        with open(path, "wb") as pickle_file:
            pickle.dump(self.buffer, pickle_file)
        print("Saved _buffer_ at path `{}`".format(path))

        path = os.path.join(logdir, "model.pt")
        torch.save(model.state_dict(), path)
        print("Saved _model_ at path `{}`".format(path))


    def train(self, n_batches, log_fn=None, log_every=1):
        losses = []
        for epoch in range(1, self.n_train_epochs + 1):
            losses.append([])
            for states, actions, _, deltas in self.buffer.train_batches(self.batch_size):

                self.model.train()
                self.optim.zero_grad()
                loss = self.model.loss(states, actions, deltas)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip_norm, norm_type=2)
                self.optim.step()

                losses[epoch - 1].append(loss.item())

            if log_fn is not None and epoch % log_every == 0:
                log_fn(epoch, self._get_avg_loss(losses, n_batches, epoch))

        return self._get_avg_loss(losses, n_batches, epoch)

    def reset_models(self):
        self.model.reset_parameters()
        self.params = list(self.model.parameters())
        self.optim = torch.optim.Adam(self.params, lr=self.learning_rate, eps=self.epsilon)

    def _get_avg_loss(self, losses, n_batches, epoch):
        epoch_loss = [sum(loss) / n_batches for loss in losses]
        return sum(epoch_loss) / epoch
