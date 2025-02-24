# movie-rec

CSEN 169 Project II: Movie Recommendation

## Report

Run with:

```bash
uv run src.py
```

Baseline performances:

```bash
cosine similarity test loss: 1.0093
pearson test loss: 1.2220
pearson iuf test loss: 1.0605
pearson iuf test loss with case mod: 1.0679
item-based test loss: 1.0825
NN test loss: 1.3804
```

Random hyperparameter sweep (10 iterations, 5 epochs each):

```bash
best_params = {
    'bs': 128,
    'n_act_max': 512,
    'dropout_rate': 0.7,
    'lr': 0.001,
    'wd': 0.01,
    'n_hidden_layers': 6
}
NN test loss: 0.9600
```

Retrained NN (10 epochs):

```bash
new_params = {
    "bs": 128,
    "n_act_max": 512,
    "dropout_rate": 0.1,
    "lr": 0.01,
    "wd": 0.01,
    "n_hidden_layers": 2,
}
NN test loss: 0.7766
```
