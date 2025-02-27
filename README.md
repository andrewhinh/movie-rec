# movie-rec

CSEN 169 Project II: Movie Recommendation

## usage

Test baseline methods:

```bash
uv run src.py --baseline
```

Run hyperparameter sweep for NN:

```bash
uv run src.py --sweep
```

Test trained NN:

```bash
uv run src.py --test
```

Write predictions to disk:

```bash
uv run src.py --write
```

## report

Baseline performances:

```bash
cosine similarity test loss: 1.0093
pearson test loss: 1.2220
pearson iuf test loss: 1.0605
pearson iuf test loss with case mod: 1.0679
item-based test loss: 1.0825
NN test loss: 1.2652
```

Random hyperparameter sweep (50 iterations, 10 epochs each):

```bash
Best parameters:
{
    'bs': 32,
    'embed_p': 0.1,
    'dropout_rate': 0.0,
    'n_act_max': 2048,
    'n_hidden_layers': 5,
    'lr': 0.01,
    'wd': 0.0001
}
Best validation loss: 0.7771
```

Best NN on test set (30 epochs):

```bash
Best parameters:
{
    "bs": 32,
    "embed_p": 0.1,
    "dropout_rate": 0.3,
    "n_act_max": 2048,
    "n_hidden_layers": 5,
    "lr": 0.01,
    "wd": 0.0001,
}
NN test loss: 0.7507
```

Results:

```bash
Hello, Hinh, Andrew
The following is the summary of your submissions:
MAE of GIVEN 5 : 0.82168313117419
MAE of GIVEN 10 : 0.749
MAE of GIVEN 20 : 0.731937879810939
OVERALL MAE : 0.76559678213758

You have already submitted 2 times.
GOOD LUCK!
```
