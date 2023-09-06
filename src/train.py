import numpy as np
import torch
import torch.nn as nn
import ray
from ray.air import session
import ray.train as train
from ray.train.torch import TorchCheckpoint, TorchTrainer
from ray.air.config import CheckpointConfig, DatasetConfig, RunConfig, ScalingConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
import typer
import data
import models
import utils
from config import MLFLOW_TRACKING_URI
from typing_extensions import Annotated

app = typer.Typer()


def train_step(ds, batch_size, model, loss_fn, optimizer):
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # reset gradients        
        z = model(batch)  # forward pass
        targets = batch['y']
        J = loss_fn(z, targets)  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss

def eval_step(ds, batch_size, model, loss_fn):
    """Eval step."""
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = batch['y']
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["y"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)

def train_loop_per_worker(config):
    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    output_size = config["output_size"]

    # Get datasets
    utils.set_seeds()
    train_ds = session.get_dataset_shard("train")
    val_ds = session.get_dataset_shard("val")

    # Model
    model =  models.LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model = train.torch.prepare_model(model)

    # Training components
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    # Training
    batch_size_per_worker = batch_size // session.get_world_size()
    for epoch in range(num_epochs):
        # Step
        train_loss = train_step(train_ds, batch_size_per_worker, model, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_ds, batch_size_per_worker, model, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint
        metrics = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, val_loss=val_loss)
        checkpoint = TorchCheckpoint.from_model(model=model)
        session.report(metrics, checkpoint=checkpoint)

@app.command()
def train_model(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment for this training workload.")] = None,
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training.")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker.")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker.")] = 0,
    dataset_location: Annotated[str, typer.Option(help="location of the dataset.")] = None,
):
    # Train loop config
    train_loop_config = {
        "dropout_p": 0.5,
        "lr": 1e-4,
        "lr_factor": 0.8,
        "lr_patience": 3,
        "num_epochs": 1,
        "batch_size": 256,
        "input_size": 2,
        "hidden_size": 32,
        "output_size": 10
    }

        # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
        _max_cpu_fraction_per_node=0.8,
    )

    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True
        )

    # Dataset config
    dataset_config = {
        "train": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
        "val": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
    }

    # Trainer
    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min")


    # Run configuration with MLflow callback
    run_config = RunConfig(
        callbacks=[mlflow_callback],
        checkpoint_config=checkpoint_config,
    )

    # Dataset
    df = data.load_data(dataset_location)
    train_df, val_df = data.split_dataset(df)

    train_ds = data.to_ray_dataset(train_df)
    val_ds = data.to_ray_dataset(val_df)

    # Preprocess
    preprocessor = data.CustomPreprocessor()
    train_ds =  preprocessor.fit_transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={"train": train_ds, "val": val_ds},
    dataset_config=dataset_config,
    preprocessor=preprocessor,
    )

    # Train
    results = trainer.fit()

    print(f"Fuck, got results: {results}")
    return results

if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()  # initialize Typer app
