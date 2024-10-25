# Default arguments for the model
device = 'cpu'
optimizer_name = 'Adam'
support_set_size = 5
num_epochs = 5
learning_rate = 0.001
threshold = 0.5
batch=32
seed=69
num_workers=0
hidden_dims = [128, 64, 32]

# Hyperparameter optimization
trial_num = 1000
n_jobs = 99
few_shot = False

# Hidden dims dictionary. Do not change predefined values. You can add new values.
hidden_dims_dict = {
    1: [128, 64],
    2: [128, 64, 32],
    3: [256, 128, 64],
    4: [256, 128, 64, 32]
}


test_embeddings_path = None # Enter the path of the test embeddings
support_set_path = None # Enter the path of the support set
embeddings_path = None # Enter the path of the embeddings