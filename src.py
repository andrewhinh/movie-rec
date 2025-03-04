import argparse
import math
import multiprocessing
import random
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_user_ratings(
    df: pd.DataFrame, user_id: int
) -> pd.Series:  # index = movie id, value = rating
    return df[df["user"] == user_id].set_index("movie")["rating"]


def calc_user_sim(
    s1: pd.Series,
    s2: pd.Series,
    method: str = "cosine",
    iuf: bool = False,
    case_mod: bool = False,
    p: float = 2.5,
    total_users: int = None,
    popularity: dict = None,
) -> float:
    assert method in ["cosine", "pearson"], f"Invalid method: {method}"

    common = s1.index.intersection(s2.index)
    if common.empty:
        return 0.0

    if method == "pearson" and iuf and total_users and popularity:
        s1 = s1.loc[common].astype(float).copy()
        s2 = s2.loc[common].astype(float).copy()
        for m in common:
            iuf_factor = math.log(total_users / popularity[m])
            s1.loc[m] *= iuf_factor
            s2.loc[m] *= iuf_factor
    else:
        s1 = s1.loc[common]
        s2 = s2.loc[common]

    if method == "cosine":
        numer = (s1 * s2).sum()
        denom = math.sqrt((s1**2).sum()) * math.sqrt((s2**2).sum())
        return numer / denom if denom != 0 else 0.0
    elif method == "pearson":
        s1_avg, s2_avg = s1.mean(), s2.mean()
        numer = ((s1 - s1_avg) * (s2 - s2_avg)).sum()
        denom = math.sqrt(((s1 - s1_avg) ** 2).sum() * ((s2 - s2_avg) ** 2).sum())
        if denom == 0:
            return 0.0
        corr = numer / denom
        if case_mod:
            corr *= abs(corr) ** (p - 1)
        return corr


def user_pred_rating(
    target_user_id: int,
    movie_id: int,
    df: pd.DataFrame,
    method: str = "cosine",
    k: int = 5,
    iuf: bool = False,
    case_mod: bool = False,
    p: float = 2.5,
) -> float:
    assert method in ["cosine", "pearson"]

    target_ratings = get_user_ratings(df, target_user_id)
    if movie_id in target_ratings.index:
        return target_ratings.loc[movie_id]

    other_users = df[df["movie"] == movie_id]["user"].unique()
    total_users = df["user"].nunique() if (iuf and method == "pearson") else None
    popularity = (
        df.groupby("movie").size().to_dict() if (iuf and method == "pearson") else None
    )

    sims = []
    for other_user in other_users:
        if other_user == target_user_id:
            continue
        other_ratings = get_user_ratings(df, other_user)
        sim = calc_user_sim(
            target_ratings,
            other_ratings,
            method=method,
            iuf=iuf,
            case_mod=case_mod,
            p=p,
            total_users=total_users,
            popularity=popularity,
        )
        sims.append((other_user, sim))
    if not sims:
        return target_ratings.mean() if not target_ratings.empty else 3.0

    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[:k]

    numer, denom = 0.0, 0.0
    for neighbor, sim in top_k:
        neighbor_rating = get_user_ratings(df, neighbor).loc[movie_id]
        numer += sim * neighbor_rating
        denom += abs(sim)
    if denom == 0:
        return target_ratings.mean() if not target_ratings.empty else 3.0

    return numer / denom


# for batch predictions
def user_pred_rating_batch(
    user_movie_pairs: pd.DataFrame,
    df: pd.DataFrame,
    method: str = "cosine",
    k: int = 5,
    iuf: bool = False,
    case_mod: bool = False,
    p: float = 2.5,
) -> torch.Tensor:
    assert method in ["cosine", "pearson"]
    predictions = []
    for user_id, movie_id in user_movie_pairs.tolist():
        r = user_pred_rating(
            user_id, movie_id, df, method=method, k=k, iuf=iuf, case_mod=case_mod, p=p
        )
        predictions.append(r)
    return torch.tensor([predictions], dtype=torch.float32).to(device)


def item_cos_sim(df: pd.DataFrame, movie1: int, movie2: int) -> float:
    df1 = df[df["movie"] == movie1][["user", "rating"]]
    df2 = df[df["movie"] == movie2][["user", "rating"]]
    common = pd.merge(df1, df2, on="user", suffixes=("_1", "_2"))
    if common.empty:
        return 0.0
    numer = (common["rating_1"] * common["rating_2"]).sum()
    denom = math.sqrt((common["rating_1"] ** 2).sum()) * math.sqrt(
        (common["rating_2"] ** 2).sum()
    )
    return numer / denom if denom != 0 else 0.0


def item_pred_rating(
    user_id: int, target_movie: int, df: pd.DataFrame, k: int = 5
) -> float:
    user_ratings = df[df["user"] == user_id]
    if target_movie in user_ratings["movie"].values:
        return user_ratings[user_ratings["movie"] == target_movie]["rating"].iloc[0]

    sims = []
    for _, row in user_ratings.iterrows():
        movie, rating = row["movie"], row["rating"]
        sim = item_cos_sim(df, target_movie, movie)
        sims.append((rating, sim))
    if not sims:
        return user_ratings["rating"].mean() if not user_ratings.empty else 3.0

    sims.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = sims[:k]
    numer, denom = 0.0, 0.0
    for rating, sim in top_neighbors:
        numer += sim * rating
        denom += abs(sim)
    if denom == 0:
        return user_ratings["rating"].mean()
    return numer / denom


def item_pred_rating_batch(
    user_movie_pairs: pd.DataFrame, df: pd.DataFrame, k: int = 5
) -> torch.Tensor:
    predictions = []
    for user_id, movie_id in user_movie_pairs.tolist():
        predictions.append(item_pred_rating(user_id, movie_id, df, k))
    return torch.tensor([predictions], dtype=torch.float32).to(device)


loss_fn = nn.L1Loss()  # equivalent to mean absolute error


def compute_loss(fn, data_loader: DataLoader) -> float:
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = fn(inputs)
            predictions = (
                predictions.squeeze()
                if predictions.dim() > targets.dim()
                else predictions
            )
            targets = (
                targets.squeeze() if targets.dim() > predictions.dim() else targets
            )
            loss = loss_fn(predictions, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(data_loader.dataset)


# random forest batch prediction
def rf_pred_rating_batch(
    batch: torch.Tensor, model: RandomForestRegressor
) -> torch.Tensor:
    return torch.tensor(model.predict(batch.cpu().numpy()), dtype=torch.float32).to(
        device
    )


# hist gradient boosting batch prediction
def hg_pred_rating_batch(
    batch: torch.Tensor, model: HistGradientBoostingRegressor
) -> torch.Tensor:
    return torch.tensor(model.predict(batch.cpu().numpy()), dtype=torch.float32).to(
        device
    )


# NN
def get_emb_sz(
    n_items: int,
) -> tuple[
    int, int
]:  #  rule of thumb for embedding size from https://docs.fast.ai/tabular.model.html#emb_sz_rule
    return n_items, int(min(600, round(1.6 * n_items**0.56)))


class NN(nn.Module):  # based on https://docs.fast.ai/tabular.model.html#tabularmodel
    def __init__(
        self,
        user_sz: tuple[int, int],
        movie_sz: tuple[int, int],
        embed_p: float,
        dropout_rate: float,
        n_act_max: int,
        n_hidden_layers: int,
        y_range: tuple[float, float],
    ):
        super().__init__()
        self.user_factors = nn.Embedding(*user_sz)
        self.movie_factors = nn.Embedding(*movie_sz)
        self.emb_drop = nn.Dropout(embed_p)

        _layers = [
            nn.BatchNorm1d(user_sz[1] + movie_sz[1]),
            nn.Dropout(dropout_rate),
            nn.Linear(user_sz[1] + movie_sz[1], n_act_max),
            nn.ReLU(),
        ]
        n_act = n_act_max
        for _ in range(n_hidden_layers):
            _layers.extend(
                [
                    nn.BatchNorm1d(n_act),
                    nn.Dropout(dropout_rate),
                    nn.Linear(n_act, n_act // 2),
                    nn.ReLU(),
                ]
            )
            n_act //= 2
        _layers.append(nn.Linear(n_act, 1))
        self.layers = nn.Sequential(*_layers)

        self.y_range = y_range

        self.init_weights()

    def init_weights(self):
        self.user_factors.weight.data.normal_(mean=0.0, std=0.01)
        self.movie_factors.weight.data.normal_(mean=0.0, std=0.01)
        for module in self.layers:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        user_emb = self.user_factors(x[:, 0])
        movie_emb = self.movie_factors(x[:, 1])
        x = torch.cat([user_emb, movie_emb], dim=1)
        x = self.emb_drop(x)
        x = self.layers(x)
        return self.y_range[0] + (self.y_range[1] - self.y_range[0]) * torch.sigmoid(x)


base_optimizer = optim.Adam


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    num_epochs: int,
    wd: float,
) -> float:
    total_steps = len(train_loader) * num_epochs
    optimizer = base_optimizer(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                predictions = model(inputs).squeeze(1)
                loss = loss_fn(predictions, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * inputs.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            ):
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs).squeeze(1)
                loss = loss_fn(predictions, targets)
                total_val_loss += loss.item() * inputs.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
    return avg_val_loss


# extend user embeddings for new user(s)
def extend_user_embeds(model: nn.Module, new_n_users: int) -> nn.Module:
    old_weight = model.user_factors.weight.data
    old_n_users = old_weight.size(0)
    if new_n_users <= old_n_users:
        return model
    new_embedding = nn.Embedding(new_n_users, old_weight.size(1)).to(device)
    new_weight = new_embedding.weight.data
    new_weight[:old_n_users] = old_weight
    avg = old_weight.mean(
        dim=0, keepdim=True
    )  # embedding for new users is the average of all existing users
    new_weight[old_n_users:] = avg.expand(new_n_users - old_n_users, -1)
    model.user_factors = new_embedding  # safe to reassign since no test-time adapted embeddings are written
    return model


# adapt existing user embedding for new data
def adapt_user_embed(
    model: nn.Module,
    test_user_index: int,
    known_movie_ids: list[int],
    known_ratings: list[float],
    lr: float,
    num_steps: int,
    wd: float,
    alpha: float = 0.5,  # 0.0 = no adaptation, 1.0 = full adaptation
) -> None:
    model.eval()

    if len(known_ratings) == 5:  # adjust hyperparams for less ratings
        lr /= 2
        num_steps *= 2

    user_emb = (
        model.user_factors.weight[test_user_index]
        .clone()
        .detach()
        .requires_grad_(True)
        .to(device)
    )
    movie_ids_tensor = torch.tensor(known_movie_ids, dtype=torch.long).to(device)
    movie_embs = model.movie_factors(movie_ids_tensor).detach()
    targets = torch.tensor(known_ratings, dtype=torch.float32).to(device)

    optimizer = base_optimizer([user_emb], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=num_steps
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for _ in range(num_steps):
        optimizer.zero_grad()
        user_emb_expanded = user_emb.unsqueeze(0).expand(movie_embs.size(0), -1)
        input_vec = torch.cat([user_emb_expanded, movie_embs], dim=1)
        with torch.amp.autocast(
            device_type=device.type, enabled=(device.type == "cuda")
        ):
            output = model.layers(input_vec).squeeze(1)
            preds = model.y_range[0] + (
                model.y_range[1] - model.y_range[0]
            ) * torch.sigmoid(output)
            loss = loss_fn(preds, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    if len(known_ratings) == 5:
        global_avg = model.user_factors.weight.mean(dim=0)
        user_emb = alpha * user_emb + (1 - alpha) * global_avg
    return user_emb.detach()  # so that we isolate each new user's updated embedding


def nn_pred_rating_batch(
    model: nn.Module,
    user_emb: torch.Tensor,
    movie_ids: list[int],
) -> list[float]:
    movie_ids_tensor = torch.tensor(movie_ids, dtype=torch.long).to(device)
    movie_embs = model.movie_factors(movie_ids_tensor)
    user_emb_expand = user_emb.unsqueeze(0).expand(movie_embs.size(0), -1)
    embs = torch.cat([user_emb_expand, movie_embs], dim=1)
    output = model.layers(embs).squeeze(1)
    preds = model.y_range[0] + (model.y_range[1] - model.y_range[0]) * torch.sigmoid(
        output
    )
    return preds.cpu().detach().numpy().tolist()


def process_test_file(
    model: nn.Module,
    input_filepath: str,
    output_filepath: str,
    trained_n_users: int,
    lr: float,
    num_steps: int,
    wd: float,
) -> None:
    with open(input_filepath, "r") as f:
        lines = f.readlines()

    output_lines = []
    current_user = None
    block_lines = []
    block_known_movie_ids = []  # 0-idx movie IDs with provided ratings
    block_known_ratings = []
    block_unknown_movie_ids = []  # 0-idx movie IDs to predict

    for line in tqdm(lines, desc="Processing lines"):
        line = line.strip()
        parts = line.split()

        # 1-idx (file) to 0-idx (for model)
        user_id = int(parts[0]) - 1
        movie_id = int(parts[1]) - 1
        rating = float(parts[2])

        # when the user changes, process the accumulated block
        if current_user is None:
            current_user = user_id
        if user_id != current_user:
            test_user_index = current_user
            if test_user_index >= model.user_factors.weight.size(0):
                model = extend_user_embeds(model, test_user_index + 1)
            if block_known_movie_ids:
                adapted_emb = adapt_user_embed(
                    model,
                    test_user_index,
                    block_known_movie_ids,
                    block_known_ratings,
                    lr,
                    num_steps,
                    wd,
                )
            else:  # otherwise, use the mean embedding of the known users
                adapted_emb = model.user_factors.weight[:trained_n_users].mean(dim=0)
            preds = (
                nn_pred_rating_batch(model, adapted_emb, block_unknown_movie_ids)
                if block_unknown_movie_ids
                else []
            )

            pred_idx = 0
            for u, m, r in block_lines:
                if r == 0 and pred_idx < len(preds):
                    r = preds[pred_idx]
                    pred_idx += 1
                    output_lines.append(
                        f"{u+1} {m+1} {max(1, min(round(r), 5))}\n"
                    )  # round to nearest int, 1 if < 1 and 5 if > 5

            # reset for the new user
            current_user = user_id
            block_lines = []
            block_known_movie_ids = []
            block_known_ratings = []
            block_unknown_movie_ids = []

        # append the current line’s data (already in 0-idx)
        block_lines.append((user_id, movie_id, rating))
        if rating != 0:
            block_known_movie_ids.append(movie_id)
            block_known_ratings.append(rating)
        else:
            block_unknown_movie_ids.append(movie_id)

    # last user block
    if block_lines:
        test_user_index = current_user
        if test_user_index >= model.user_factors.weight.size(0):
            model = extend_user_embeds(model, test_user_index + 1)
        if block_known_movie_ids:
            adapted_emb = adapt_user_embed(
                model,
                test_user_index,
                block_known_movie_ids,
                block_known_ratings,
                lr,
                num_steps,
                wd,
            )
        else:
            adapted_emb = model.user_factors.weight[:trained_n_users].mean(dim=0)
        preds = (
            nn_pred_rating_batch(model, adapted_emb, block_unknown_movie_ids)
            if block_unknown_movie_ids
            else []
        )

        pred_idx = 0
        for u, m, r in block_lines:
            if r == 0 and pred_idx < len(preds):
                r = preds[pred_idx]
                pred_idx += 1
                output_lines.append(f"{u+1} {m+1} {max(1, min(round(r), 5))}\n")

    with open(output_filepath, "w") as f:
        f.writelines(output_lines)
    print(f"Predictions written to {output_filepath}")


# main
def main(args):
    # deterministic
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # given
    n_users, n_movies = 200, 1000
    embs = get_emb_sz(n_users), get_emb_sz(n_movies)
    y_range = (
        0.5,
        5.5,
    )  # empirically, better performance is achieved when model can predict values beyond given range ([1, 5])

    # load data
    df = pd.read_csv(
        "data/train.txt", sep=" ", header=None, names=["user", "movie", "rating"]
    )

    x = torch.tensor(
        [
            (user_id - 1, movie_id - 1)
            for user_id, movie_id in zip(df["user"], df["movie"])
        ],
        dtype=torch.long,
    )  # adjusting indices for 0-based embedding
    y = torch.tensor(df["rating"], dtype=torch.float32)

    # train-val-test split
    train_size, val_size, test_size = 0.9, 0.05, 0.05
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_size / (train_size + val_size),
        random_state=seed,
    )

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )

    # baseline methods eval (using batch functions)
    if args.baseline:
        print(
            f"cosine similarity test MAE: {compute_loss(partial(user_pred_rating_batch, df=df, method='cosine'), test_loader):.4f}"
        )
        print(
            f"pearson test MAE: {compute_loss(partial(user_pred_rating_batch, df=df, method='pearson'), test_loader):.4f}"
        )
        print(
            f"pearson iuf test MAE: {compute_loss(partial(user_pred_rating_batch, df=df, method='pearson', iuf=True), test_loader):.4f}"
        )
        print(
            f"pearson iuf test MAE with case mod: {compute_loss(partial(user_pred_rating_batch, df=df, method='pearson', iuf=True, case_mod=True), test_loader):.4f}"
        )
        print(
            f"item-based test MAE: {compute_loss(partial(item_pred_rating_batch, df=df), test_loader):.4f}"
        )

        # rf
        n_estimators = 100
        max_features = 0.5
        min_samples_leaf = 5
        rf = RandomForestRegressor(
            n_jobs=-1,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            oob_score=True,
            random_state=seed,
        )
        rf.fit(x_train.numpy(), y_train.numpy())
        print(
            f"random forest test MAE: {compute_loss(partial(rf_pred_rating_batch, model=rf), test_loader):.4f}"
        )

        # hg
        lr = 0.1
        max_iter = 200
        max_features = 0.5
        min_samples_leaf = 20
        max_leaf_nodes = 31
        l2_regularization = 0.0
        hg = HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
        )
        hg.fit(x_train.numpy(), y_train.numpy())
        print(
            f"hist gradient boosting test MAE: {compute_loss(partial(hg_pred_rating_batch, model=hg), test_loader):.4f}"
        )

        # nn
        embed_p = 0.0
        dropout_rate = 0.0
        n_hidden_layers = 2
        n_act_max = 512
        model = NN(
            *embs, embed_p, dropout_rate, n_act_max, n_hidden_layers, y_range
        ).to(device)
        model.eval()
        print(f"NN test MAE: {compute_loss(model, test_loader):.4f}")

    # hyperparam sweep
    if args.sweep:
        param_grid = {
            "bs": [2**i for i in range(5, 11)],  # 32 to 1024
            "embed_p": [i / 10 for i in range(6)],  # 0.0 to 0.5
            "dropout_rate": [i / 10 for i in range(6)],  # 0.0 to 0.5
            "n_act_max": [2**i for i in range(9, 14)],  # 256 to 8192
            "n_hidden_layers": [i for i in range(2, 7)],  # 2 to 6
            "lr": [10**i for i in range(-4, 0)],  # 1e-4 to 0.1
            "wd": [10**i for i in range(-4, 0)],  # 1e-4 to 0.1
        }
        num_epochs = 10
        n_sweep_it = 50

        best_val_loss, best_params = float("inf"), {}
        for _ in tqdm(range(n_sweep_it), desc="Hyperparameter Sweep"):
            bs = random.choice(param_grid["bs"])
            embed_p = random.choice(param_grid["embed_p"])
            dropout_rate = random.choice(param_grid["dropout_rate"])
            n_act_max = random.choice(param_grid["n_act_max"])
            n_hidden_layers = random.choice(param_grid["n_hidden_layers"])
            lr = random.choice(param_grid["lr"])
            wd = random.choice(param_grid["wd"])

            train_loader = DataLoader(
                train_dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=multiprocessing.cpu_count(),
                pin_memory=True,
            )

            model = NN(
                *embs, embed_p, dropout_rate, n_act_max, n_hidden_layers, y_range
            ).to(device)
            val_loss = train_loop(model, train_loader, val_loader, lr, num_epochs, wd)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {
                    "bs": bs,
                    "embed_p": embed_p,
                    "dropout_rate": dropout_rate,
                    "n_act_max": n_act_max,
                    "n_hidden_layers": n_hidden_layers,
                    "lr": lr,
                    "wd": wd,
                }

        print(f"Best parameters: {best_params}")
        print(f"Best validation MAE: {best_val_loss:.4f}")

    if args.test or args.write:
        best_params = {
            "bs": 128,
            "embed_p": 0.1,
            "dropout_rate": 0.3,
            "n_act_max": 8192,
            "n_hidden_layers": 4,
            "lr": 0.0001,
            "wd": 0.01,
        }
        model = NN(
            *embs,
            best_params["embed_p"],
            best_params["dropout_rate"],
            best_params["n_act_max"],
            best_params["n_hidden_layers"],
            y_range,
        ).to(device)
        train_loader = DataLoader(
            train_dataset,
            batch_size=best_params["bs"],
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )
        num_epochs = 10
        train_loop(
            model,
            train_loader,
            val_loader,
            best_params["lr"],
            num_epochs,
            best_params["wd"],
        )
        model.eval()

        # evaluate best model on test set
        if args.test:
            print(f"NN test MAE: {compute_loss(model, test_loader):.4f}")

        # write to test files
        if args.write:
            test_files = ["data/test5.txt", "data/test10.txt", "data/test20.txt"]
            num_steps = 100  # number of steps to adapt user embedding
            for test_file in test_files:
                output_file = test_file.replace("test", "pred_test")
                process_test_file(
                    model,
                    test_file,
                    output_file,
                    n_users,
                    best_params["lr"],
                    num_steps,
                    best_params["wd"],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    assert any(vars(args).values()), "At least one argument must be provided"
    main(args)
