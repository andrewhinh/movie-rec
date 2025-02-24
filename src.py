import math
import multiprocessing
import random

# from functools import partial
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.L1Loss()
base_optimizer = optim.Adam


def get_user_ratings(
    df: pd.DataFrame, user_id: int
) -> pd.Series:  # index = movie id, value = rating
    return df[df["user"] == user_id].set_index("movie")["rating"]


# sim between users
def cosine_similarity(s1: pd.Series, s2: pd.Series) -> float:
    common = s1.index.intersection(s2.index)
    if common.empty:
        return 0.0
    a = s1.loc[common]
    b = s2.loc[common]
    numer = (a * b).sum()
    denom = math.sqrt((a**2).sum()) * math.sqrt((b**2).sum())
    return numer / denom if denom != 0 else 0.0


def pearson_corr(s1: pd.Series, s2: pd.Series) -> float:
    common = s1.index.intersection(s2.index)
    if common.empty:
        return 0.0
    a = s1.loc[common]
    b = s2.loc[common]
    a_avg, b_avg = a.mean(), b.mean()
    numer = ((a - a_avg) * (b - b_avg)).sum()
    denom = math.sqrt(((a - a_avg) ** 2).sum() * ((b - b_avg) ** 2).sum())
    return numer / denom if denom != 0 else 0.0


# user-based
def user_pred_rating(
    target_user_id: int,
    movie_id: int,
    df: pd.DataFrame,  # to get similar users and ratings
    similarity_func=cosine_similarity,
    k: int = 5,
) -> float:
    target_ratings = get_user_ratings(df, target_user_id)
    if movie_id in target_ratings.index:
        return target_ratings.loc[movie_id]

    other_users = df[df["movie"] == movie_id]["user"].unique()
    sims = []
    for other_user in other_users:
        if other_user == target_user_id:
            continue
        other_ratings = get_user_ratings(df, other_user)
        sim = similarity_func(target_ratings, other_ratings)
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


def user_pred_rating_batch(
    user_movie_pairs: pd.DataFrame,
    df: pd.DataFrame,
    similarity_func=cosine_similarity,
    k: int = 5,
) -> torch.Tensor:
    predictions = []
    for user_id, movie_id in user_movie_pairs.tolist():
        predictions.append(user_pred_rating(user_id, movie_id, df, similarity_func, k))
    return torch.tensor([predictions], dtype=torch.float32).to(device)  # shape (1, B)


# iuf-weighted pearson
def pearson_correlation_iuf(
    s1: pd.Series,
    s2: pd.Series,
    popularity: dict,
    total_users: int,
    case_mod: bool,
    p: float,
) -> float:
    common = s1.index.intersection(s2.index)
    if common.empty:
        return 0.0
    s1_weighted = s1.loc[common].copy()
    s2_weighted = s2.loc[common].copy()
    for m in common:
        iuf_factor = math.log(total_users / popularity[m])
        s1_weighted = s1_weighted.astype(float)
        s2_weighted = s2_weighted.astype(float)
        s1_weighted.loc[m] *= iuf_factor
        s2_weighted.loc[m] *= iuf_factor
    s1_avg, s2_avg = s1_weighted.mean(), s2_weighted.mean()
    numer = ((s1_weighted - s1_avg) * (s2_weighted - s2_avg)).sum()
    denom = math.sqrt(
        ((s1_weighted - s1_avg) ** 2).sum() * ((s2_weighted - s2_avg) ** 2).sum()
    )
    if denom == 0:
        return 0.0
    pearson = numer / denom
    return pearson * (abs(pearson) ** (p - 1)) if case_mod else pearson


def iuf_case_pred_rating(
    user_id: int,
    movie_id: int,
    df: pd.DataFrame,
    case_mod: bool = True,
    k: int = 5,
    p: float = 2.5,
) -> float:
    total_users = df["user"].nunique()
    popularity = df.groupby("movie").size().to_dict()
    target_ratings = get_user_ratings(df, user_id)
    if movie_id in target_ratings.index:
        return target_ratings.loc[movie_id]

    other_users = df[df["movie"] == movie_id]["user"].unique()
    sims = []
    for other_user in other_users:
        if other_user == user_id:
            continue
        other_ratings = get_user_ratings(df, other_user)
        sim = pearson_correlation_iuf(
            target_ratings, other_ratings, popularity, total_users, case_mod, p
        )
        sims.append((other_user, sim))

    if not sims:
        return target_ratings.mean() if not target_ratings.empty else 3.0

    sims.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = sims[:k]
    numer, denom = 0.0, 0.0
    for nbr, sim in top_neighbors:
        nbr_rating = get_user_ratings(df, nbr).loc[movie_id]
        numer += sim * nbr_rating
        denom += abs(sim)
    if denom == 0:
        return target_ratings.mean() if not target_ratings.empty else 3.0
    return numer / denom


def iuf_case_pred_rating_batch(
    user_movie_pairs: pd.DataFrame,
    df: pd.DataFrame,
    case_mod: bool = True,
    k: int = 5,
    p: float = 2.5,
) -> torch.Tensor:
    predictions = []
    for user_id, movie_id in user_movie_pairs.tolist():
        predictions.append(iuf_case_pred_rating(user_id, movie_id, df, case_mod, k, p))
    return torch.tensor([predictions], dtype=torch.float32).to(device)


# item-based
def item_cosine_similarity(df: pd.DataFrame, movie1: int, movie2: int) -> float:
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
        sim = item_cosine_similarity(df, target_movie, movie)
        sims.append((movie, sim, rating))

    if not sims:
        return user_ratings["rating"].mean() if not user_ratings.empty else 3.0

    sims.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = sims[:k]
    numer = sum(sim * rating for (_, sim, rating) in top_neighbors)
    denom = sum(abs(sim) for (_, sim, _) in top_neighbors)
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


# NN
def get_emb_sz(n_items: int) -> tuple[int, int]:
    "Rule of thumb for embedding size."
    return n_items, int(min(600, round(1.6 * n_items**0.56)))


class NN(nn.Module):
    def __init__(
        self, user_sz, movie_sz, y_range, n_hidden_layers, n_act_max, dropout_rate
    ):
        super().__init__()
        self.user_factors = nn.Embedding(*user_sz)
        self.movie_factors = nn.Embedding(*movie_sz)
        layers = [
            nn.Linear(user_sz[1] + movie_sz[1], n_act_max),
            nn.BatchNorm1d(n_act_max),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]
        for i in range(1, n_hidden_layers):
            n_act = n_act_max // (i + 1)
            layers.extend(
                [
                    nn.Linear(n_act_max // i, n_act),
                    nn.BatchNorm1d(n_act),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        layers.append(nn.Linear(n_act_max // n_hidden_layers, 1))
        self.layers = nn.Sequential(*layers)
        self.y_range = y_range

    def forward(self, x):
        embs = self.user_factors(x[:, 0]), self.movie_factors(x[:, 1])
        x = self.layers(torch.cat(embs, dim=1))
        return self.y_range[0] + (self.y_range[1] - self.y_range[0]) * torch.sigmoid(x)


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


def compute_test_loss(fn, test_loader: DataLoader) -> float:
    total_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
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
            total_test_loss += loss.item() * inputs.size(0)
    return total_test_loss / len(test_loader.dataset)


def extend_model_user_embeddings(model: nn.Module, new_n_users: int) -> nn.Module:
    old_weight = model.user_factors.weight.data
    old_n_users = old_weight.size(0)
    if new_n_users <= old_n_users:
        return model
    new_embedding = nn.Embedding(new_n_users, old_weight.size(1)).to(device)
    new_weight = new_embedding.weight.data
    new_weight[:old_n_users] = old_weight
    avg = old_weight.mean(dim=0, keepdim=True)
    new_weight[old_n_users:] = avg.expand(new_n_users - old_n_users, -1)
    model.user_factors = new_embedding
    return model


def adapt_user_embedding(
    model: nn.Module,
    test_user_index: int,
    known_movie_ids: list[int],
    known_ratings: list[float],
    lr: float,
    num_steps: int,
    wd: float,
) -> None:
    model.eval()
    user_emb = (
        model.user_factors.weight[test_user_index]
        .clone()
        .detach()
        .requires_grad_(True)
        .to(device)
    )
    movie_ids_tensor = torch.tensor(known_movie_ids, dtype=torch.long).to(device)
    targets = torch.tensor(known_ratings, dtype=torch.float32).to(device)
    movie_embs = model.movie_factors(movie_ids_tensor).detach()

    optimizer = base_optimizer([user_emb], lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for _ in tqdm(range(num_steps), desc="Adapting user embedding"):
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

    return user_emb.detach()


def predict_for_test_user(
    model: nn.Module,
    user_emb: torch.Tensor,
    movie_ids: list[int],
) -> list[float]:
    movie_ids_tensor = torch.tensor(movie_ids, dtype=torch.long).to(device)
    movie_embs = model.movie_factors(movie_ids_tensor)
    user_emb_expanded = user_emb.unsqueeze(0).expand(movie_embs.size(0), -1)
    input_vec = torch.cat([user_emb_expanded, movie_embs], dim=1)
    output = model.layers(input_vec).squeeze(1)
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
    block_known_movie_ids = []  # 0-indexed movie IDs with provided ratings
    block_known_ratings = []
    block_unknown_movie_ids = []  # 0-indexed movie IDs to predict

    for line in lines:
        line = line.strip()
        parts = line.split()

        # 1-idx (file) to 0-idx (internal)
        user_id = int(parts[0]) - 1
        movie_id = int(parts[1]) - 1
        rating = float(parts[2])

        # When the user changes, process the accumulated block.
        if current_user is None:
            current_user = user_id
        if user_id != current_user:
            test_user_index = current_user
            if test_user_index >= model.user_factors.weight.size(0):
                model = extend_model_user_embeddings(model, test_user_index + 1)
            if block_known_movie_ids:
                adapted_emb = adapt_user_embedding(
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
                predict_for_test_user(model, adapted_emb, block_unknown_movie_ids)
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

        # Append the current line’s data (already in 0-index).
        block_lines.append((user_id, movie_id, rating))
        if rating != 0:
            block_known_movie_ids.append(movie_id)
            block_known_ratings.append(rating)
        else:
            block_unknown_movie_ids.append(movie_id)

    # Process the last user block.
    if block_lines:
        test_user_index = current_user
        if test_user_index >= model.user_factors.weight.size(0):
            model = extend_model_user_embeddings(model, test_user_index + 1)
        if block_known_movie_ids:
            adapted_emb = adapt_user_embedding(
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
            predict_for_test_user(model, adapted_emb, block_unknown_movie_ids)
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
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_users, n_movies = 200, 1000
    embs = get_emb_sz(n_users), get_emb_sz(n_movies)
    y_range = (0, 5.5)

    df = pd.read_csv(
        "data/train.txt", sep=" ", header=None, names=["user", "movie", "rating"]
    )

    # sanity checks
    assert not df.isnull().values.any()
    assert df["user"].max() == n_users
    assert df["movie"].max() == n_movies

    # adjusting indices for 0-based embedding
    x = torch.tensor(
        [
            (user_id - 1, movie_id - 1)
            for user_id, movie_id in zip(df["user"], df["movie"])
        ],
        dtype=torch.long,
    )
    y = torch.tensor(df["rating"], dtype=torch.float32)

    # train-val-test split
    train_size, val_size, test_size = 0.8, 0.1, 0.1
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

    # baseline methods eval (using batch functions)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    # print(
    #     f"cosine similarity test loss: {compute_test_loss(partial(user_pred_rating_batch, df=df, similarity_func=cosine_similarity), test_loader):.4f}"
    # )
    # print(
    #     f"pearson test loss: {compute_test_loss(partial(user_pred_rating_batch, df=df, similarity_func=pearson_corr), test_loader):.4f}"
    # )
    # print(
    #     f"pearson iuf test loss: {compute_test_loss(partial(iuf_case_pred_rating_batch, df=df, case_mod=False), test_loader):.4f}"
    # )
    # print(
    #     f"pearson iuf test loss with case mod: {compute_test_loss(partial(iuf_case_pred_rating_batch, df=df, case_mod=True), test_loader):.4f}"
    # )
    # print(
    #     f"item-based test loss: {compute_test_loss(partial(item_pred_rating_batch, df=df), test_loader):.4f}"
    # )

    # # nn baseline
    # n_hidden_layers = 2
    # n_act_max = 512
    # dropout_rate = 0.0
    # model = NN(*embs, y_range, n_hidden_layers, n_act_max, dropout_rate).to(device)
    # model.eval()
    # print(f"NN test loss: {compute_test_loss(model, test_loader):.4f}")

    # # hyperparam sweep
    # param_grid = {
    #     "bs": [2**i for i in range(1, 10)],  # 2 to 1024
    #     "n_act_max": [2**i for i in range(8, 10)],  # 256 to 1024
    #     "dropout_rate": [i / 10 for i in range(10)],  # 0.0 to 1.0
    #     "lr": [10**i for i in range(-8, -1)],  # 1e-8 to 0.1
    #     "wd": [10**i for i in range(-8, -1)],  # 1e-8 to 0.1
    #     "n_hidden_layers": [i for i in range(1, 10)],  # 1 to 10
    # }
    # num_epochs = 10
    # n_sweep_it = 50

    # best_val_loss, best_params = float("inf"), {}
    # for _ in tqdm(range(n_sweep_it), desc="Hyperparameter Sweep"):
    #     bs = random.choice(param_grid["bs"])
    #     n_act_max = random.choice(param_grid["n_act_max"])
    #     dropout_rate = random.choice(param_grid["dropout_rate"])
    #     lr = random.choice(param_grid["lr"])
    #     wd = random.choice(param_grid["wd"])
    #     n_hidden_layers = random.choice(param_grid["n_hidden_layers"])

    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=bs,
    #         shuffle=True,
    #         num_workers=multiprocessing.cpu_count(),
    #         pin_memory=True,
    #     )
    #     val_loader = DataLoader(
    #         val_dataset,
    #         batch_size=bs,
    #         shuffle=False,
    #         num_workers=multiprocessing.cpu_count(),
    #         pin_memory=True,
    #     )

    #     embs = get_emb_sz(n_users), get_emb_sz(n_movies)
    #     model = NN(*embs, y_range, n_hidden_layers, n_act_max, dropout_rate).to(device)
    #     val_loss = train_loop(model, train_loader, val_loader, lr, num_epochs, wd)

    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         best_params = {
    #             "bs": bs,
    #             "n_act_max": n_act_max,
    #             "dropout_rate": dropout_rate,
    #             "lr": lr,
    #             "wd": wd,
    #             "n_hidden_layers": n_hidden_layers,
    #         }

    # print(f"Best parameters: {best_params}")
    # print(f"Best validation loss: {best_val_loss:.4f}")

    # load best model and evaluate on test set
    best_params = {
        "bs": 128,
        "n_act_max": 512,
        "dropout_rate": 0.1,
        "lr": 0.01,
        "wd": 0.01,
        "n_hidden_layers": 2,
    }
    num_epochs = 10

    model = NN(
        *embs,
        y_range,
        best_params["n_hidden_layers"],
        best_params["n_act_max"],
        best_params["dropout_rate"],
    ).to(device)
    optimizer = base_optimizer(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["wd"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params["bs"],
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_params["bs"],
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    train_loop(
        model,
        train_loader,
        val_loader,
        best_params["lr"],
        num_epochs,
        best_params["wd"],
    )
    model.eval()
    print(f"NN test loss: {compute_test_loss(model, test_loader):.4f}")

    ## save model
    torch.save(model.state_dict(), "data/model.pt")

    ## write to test files
    test_n_users = 300  # total users after adding test users
    num_steps = 100  # number of steps to adapt user embedding
    model.load_state_dict(torch.load("data/model.pt", map_location=device))
    model = extend_model_user_embeddings(model, test_n_users)
    model.eval()
    test_files = ["data/test5.txt", "data/test10.txt", "data/test20.txt"]
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
