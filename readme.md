````markdown
# Hyperparameter Tuning for Optimizers on KMNIST

This project tunes hyperparameters for different optimizers (Adam, AdamW, and RMSprop) on the KMNIST dataset using PyTorch. It uses K-Fold cross-validation to find the best hyperparameters.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shreynpatel23/AIGC_5500_midterm_project.git
   cd AIGC_5500_midterm_project
   ```
````

2. Create a virtual environment (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Training

To start training with different optimizers, run:

```bash
python main.py
```

This will train models using Adam, AdamW, and RMSprop and save results in CSV files.

### Viewing Training Logs

To check training progress in TensorBoard:

```bash
tensorboard --logdir=logs/
```

Then open `http://localhost:6006/` in your browser.

## Hyperparameters

Each optimizer has different hyperparameters tuned during training:

- **Adam**: Learning rate, betas, weight decay
- **RMSprop**: Learning rate, alpha
- **AdamW**: Learning rate, betas, weight decay

## Results

Results are saved in CSV files, each file contains the following columns:

- `adam_training_results.csv`: learning_rate,betas,weight_decay,avg_fold_acc,total_time (s)
- `adam_w_training_results.csv`: learning_rate,betas,weight_decay,avg_fold_acc,total_time (s)
- `rmsprop_results.csv`: learning_rate,alpha,avg_fold_acc,total_time (s)
