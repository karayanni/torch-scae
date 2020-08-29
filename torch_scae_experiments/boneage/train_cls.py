from torch_scae_experiments.mnist.train import SCAEMNIST
from torch_scae_experiments.mnist.hparams import model_params
from torch_scae.factory import make_config
from argparse import Namespace
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torch import cuda
import pathlib

if cuda.is_available():
    cuda.set_device(0)
    print(cuda.current_device())

#Model Initialization
model_config = make_config(**model_params)
training_hparams = dict(
    data_dir=str(pathlib.Path.home() / 'torch-datasets'),
    gpus=1,
    batch_size=16,
    num_workers=0,
    max_epochs=100,
    learning_rate=1e-4,
    optimizer_type='RMSprop',
    use_lr_scheduler=True,
    lr_scheduler_decay_rate=0.997,
    model_config=model_config
)
scaemnist = SCAEMNIST(Namespace(**training_hparams))

#Data preparation
data_dir = training_hparams['data_dir']
# train and validation datasets
mnist_train = MNIST(data_dir, train=True, download=True, transform=scaemnist.make_transforms())
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
# assign to use in data loaders
train_dataset = mnist_train
val_dataset = mnist_val

train_dl =DataLoader(train_dataset,
                     batch_size=training_hparams['batch_size'],
                     num_workers=training_hparams['num_workers'])
val_dl = DataLoader(val_dataset,
                    batch_size=training_hparams['batch_size'],
                    num_workers=training_hparams['num_workers'])
train_out = []
val_out = []

cuda.reset_max_memory_cached()
cuda.reset_max_memory_allocated()
cuda.reset_accumulated_memory_stats()

#Training
for epoch in range(training_hparams['max_epochs']):
    for batch_idx, batch in enumerate(train_dl):
        train_batch_out = scaemnist.training_step(batch, batch_idx)
        train_out.append(train_batch_out)

    #Check first image and label in train epoch
    res_train = train_out[0]['result']
    pred_train = train_out[0]['prediction']

    for batch_idx, batch in enumerate(val_dl):
        val_batch_out = scaemnist.validation_step(batch, batch_idx)
        val_out.append(val_batch_out)

    # Check first image and label in test epoch
    res_val = val_out[0]['result']
    pred_val = val_out[0]['prediction']

    n_to_show = 8
    print(res_train.image.cpu()[:n_to_show], pred_train[:n_to_show])
    print(res_val.image.cpu()[:n_to_show], pred_val[:n_to_show])

