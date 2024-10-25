from experiment import run_experiment
import args

def objective(trial):
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    num_epochs = trial.suggest_int("num_epochs", 3, 15)
    opt_name = trial.suggest_categorical("opt_name", ["Adam", "SGD"])
    support_set_size = trial.suggest_categorical("support_set_size", [5, 10, 20, 30, 50])
    hidden_dims_key = trial.suggest_categorical("hidden_dims_key", args.hidden_dims_dict.keys())

    batch = args.batch

    hidden_dims = args.hidden_dims_dict[hidden_dims_key]

    model, results = run_experiment(
        num_epochs=num_epochs,
        learning_rate=lr,
        threshold=args.threshold,
        seed=args.seed,
        batch_size=batch,
        test_embeddings_path=args.test_embeddings_path,
        support_set_embeddings_path=args.support_set_path,
        train_embeddings_path=args.embeddings_path,
        device=args.device,
        num_workers=args.num_workers,
        optimizer_name=opt_name,
        support_set_size=support_set_size,
        hidden_dims=hidden_dims,
        save_results=False,
        test_few_shot=args.few_shot
    )

    return results["accuracy"]