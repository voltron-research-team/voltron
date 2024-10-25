import torch
from dataset import SiameseDataset
from torch.utils.data import DataLoader
from model import SiameseNetwork, save_model
import torch.nn as nn
from train import train_siamese_network
from test import test_siamese_network
import optuna
import datetime
import args


def run_experiment(num_epochs, 
               learning_rate, 
               threshold,
               seed,
               batch_size, 
               test_embeddings_path, 
               support_set_embeddings_path, 
               train_embeddings_path,
               device,
               num_workers,
               optimizer_name,
               support_set_size,
               hidden_dims,
               save_results=False,
               test_few_shot=True
               ):
    
    '''
    Run the experiment with the given hyperparameters
    Returns the trained model and the results
    '''

    print(f"Running experiment with the following hyperparameters: \n"
            f"num_epochs: {num_epochs}, \n"
            f"learning_rate: {learning_rate}, \n"
            f"threshold: {threshold}, \n"
            f"seed: {seed}, \n"
            f"batch_size: {batch_size}, \n"
            f"test_embeddings_path: {test_embeddings_path}, \n"
            f"support_set_embeddings_path: {support_set_embeddings_path}, \n"
            f"train_embeddings_path: {train_embeddings_path}, \n"
            f"device: {device}, \n"
            f"num_workers: {num_workers}, \n"
            f"optimizer_name: {optimizer_name}, \n"
            f"support_set_size: {support_set_size}, \n"
            f"hidden_dims: {hidden_dims}, \n",
            f"save_results: {save_results}, \n",
            f"test_few_shot: {test_few_shot}"
    )
    
    training_embeddings = torch.load(train_embeddings_path, map_location=device)
    siamese_dataset = SiameseDataset(
        embeddings=training_embeddings,
        seed=seed
    )
    siamese_loader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    embedding_dim = training_embeddings[0][0].view(-1).size(0)
    snn_model = SiameseNetwork(embedding_dim, hidden_dims=hidden_dims).to(device)
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(snn_model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(snn_model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.BCELoss()

    train_siamese_network(
        model=snn_model, 
        data_loader=siamese_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device, 
        threshold=threshold, 
        num_epochs=num_epochs,
        test_embeddings_path=test_embeddings_path,
        support_set_path=support_set_embeddings_path,
        batch_size=batch_size,
        num_workers=num_workers,
        support_set_size=support_set_size,
    )

    average_loss, accuracy, precision, recall, f1= test_siamese_network(
        model=snn_model, 
        test_embeddings_path=test_embeddings_path, 
        support_set_path=support_set_embeddings_path, 
        criterion=criterion, 
        device=device, 
        support_set_size=support_set_size, 
        batch_size=batch_size, 
        threshold=threshold,
        num_workers=num_workers,
        save_results=save_results,
    )

    if test_few_shot:
        average_loss, accuracy, precision, recall, f1 = test_siamese_network(
            model=snn_model, 
            test_embeddings_path=test_embeddings_path, 
            support_set_path=support_set_embeddings_path, 
            criterion=criterion, 
            device=device, 
            support_set_size=support_set_size, 
            batch_size=batch_size, 
            threshold=threshold,
            num_workers=num_workers,
            save_results=save_results,
            few_shot=True
        )

        results = {
            "average_loss": average_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        return snn_model, results

    results = {
        "average_loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return snn_model, results


def experiment_with_args():
    model, results = run_experiment(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        seed=args.seed,
        batch_size=args.batch,
        test_embeddings_path=args.test_embeddings_path,
        support_set_embeddings_path=args.support_set_path,
        train_embeddings_path=args.embeddings_path,
        device=args.device,
        num_workers=args.num_workers,
        optimizer_name=args.optimizer_name,
        support_set_size=args.support_set_size,
        hidden_dims=args.hidden_dims,
        save_results=True,
        test_few_shot=args.few_shot
    )

    save_model(model)

def experiment_with_params(params):

    # if not exist best_params, use default args

    num_epochs = params.get("num_epochs", args.num_epochs)
    lr = params.get("lr", args.learning_rate)
    opt_name = params.get("opt_name", args.optimizer_name)
    support_set_size = params.get("support_set_size", args.support_set_size)
    hidden_dims_key = params.get("hidden_dims_key", 1)

    hidden_dims = args.hidden_dims_dict[hidden_dims_key]

    model, result = run_experiment(
        num_epochs=num_epochs,
        learning_rate=lr,
        threshold=args.threshold,
        seed=args.seed,
        batch_size=args.batch,
        test_embeddings_path=args.test_embeddings_path,
        support_set_embeddings_path=args.support_set_path,
        train_embeddings_path=args.embeddings_path,
        device=args.device,
        num_workers=args.num_workers,
        optimizer_name=opt_name,
        support_set_size=support_set_size,
        hidden_dims=hidden_dims,
        save_results=True,
        test_few_shot=args.few_shot
    )

    save_model(model)

def experiment_with_hyperparameter_optimization():
    from hyperparameter_optimization import objective

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    study_name = f"accuracy_{timestamp}"

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///db.sqlite3",
                                study_name=study_name,
                                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=20, reduction_factor=3))

    study.optimize(objective, n_trials=args.trial_num, n_jobs=args.n_jobs)

    best_params = study.best_params
    best_trial = study.best_trial
    best_value = best_trial.value

    experiment_with_params(best_params)