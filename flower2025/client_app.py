
"""flower2025: A Flower / PyTorch app.

Mhd Malek Kisanieh
Samir Akhalil
"""
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower2025.task import Net, load_data
from flower2025.task import test as test_fn
from flower2025.task import train as train_fn

import torch.nn.functional as F

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    use_iid = context.run_config.get("use-iid", True)
    trainloader, _ = load_data(partition_id, num_partitions, use_iid=use_iid)

    # Check if this client is an attacker
    attacker_ids_str = context.run_config.get("attacker-ids", "none")
    if attacker_ids_str and attacker_ids_str.strip() and attacker_ids_str.lower() != "none":
        attacker_ids = [int(x.strip()) for x in attacker_ids_str.split(",")]
    else:
        attacker_ids = []
    is_attacker = partition_id in attacker_ids
    attack_type = context.run_config.get("attack-type", "label_flipping")

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        is_attacker=is_attacker,
        attack_type=attack_type,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "is_attacker": float(is_attacker),  # Log if this client is an attacker
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    use_iid = context.run_config.get("use-iid", True)
    _, valloader = load_data(partition_id, num_partitions, use_iid=use_iid)

    # Call the evaluation function
    eval_loss, eval_acc, kappa, f1, roc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "kappa": kappa,
        "f1": f1,
        "roc": roc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

