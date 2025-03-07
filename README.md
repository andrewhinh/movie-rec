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

Test trained neural network (ensure model hyperparameters are updated in the code under the section `if args.test or args.write:`):

```bash
uv run src.py --test
```

Save predictions to a file (ensure model hyperparameters are set correctly in the code before proceeding):

```bash
uv run src.py --write
```

## report

First, we randomly split the data into training, validation, and test sets of sizes **80%, 10%, and 10%** respectively.

The scikit-learn implementations of the random forest and histogram gradient boosting algorithms are used to compare the baseline user and item based approaches with machine learning models.

A PyTorch implementation using a **neural network** is also proposed. The model is based on the idea that users and movies can be represented by embeddings of arbitary size, and that we can use gradient descent to find the optimal embedding vectors. Simple approaches include computing a dot product between the user and movie embeddings, or even doing so with a learned bias for each user and movie, and adjusting the embedding vectors using gradient descent. However, the model presented here extends this idea by concatenating the user and movie embeddings and passing them through a **multilayer perceptron** (where embeddings and linear layers are initialized with values sampled from a normal distribution with mean 0 and standard deviation 0.01, and biases are initialized to 0) so that the model may learn more complex representations of users, movies, and interactions between them. Then, a **sigmoid activation function** is applied to ensure that the predicted ratings are bounded between 1 and 5.

Then, we assess the performance of the user and item-based algorithms in addition to various machine learning models on the **test set**. The results are as follows:

| Method                    | MAE    |
| ------------------------- | ------ |
| Cosine Similarity         | 1.0093 |
| Pearson                   | 1.2220 |
| Pearson IUF               | 1.0605 |
| Pearson IUF with Case Mod | 1.0679 |
| Item-based                | 1.0825 |
| Random Forest             | 0.8262 |
| Hist Gradient Boosting    | 0.8185 |
| NN                        | 1.0418 |

The baseline results show that traditional methods using a nearest neighbors approach, although simple to implement, have relatively high MAE due to limitations in the ability to capture complex interactions. The machine learning models achieve lower MAE because they can capture more complex relationships than just cosine similarity or pearson correlation. However, they can still be limited in their ability to capture complex interactions due to the geometry of their decision boundaries. The untrained Neural Network has a similar MAE to the traditional methods, which is expected since the model has yet to be trained. When trained, it can learn even more complex representations and interactions, which may lead to improved performance. However, it is computational intensive to train and its performance is sensitive to hyperparameter choices.

To train the neural network, we first run **random search for 100 iterations** to find appropriate hyperparameters. For each iteration, the model is trained for **10 epochs**, and its performance on the **validation set** determines whether it is the best model so far. The best set of hyperparameters is as follows:

| Hyperparameter     | Value |
| ------------------ | ----- |
| Batch Size         | 64    |
| Embed Dropout Rate | 0.0   |
| Dropout Rate       | 0.2   |
| Act Max            | 2048  |
| Hidden Layers      | 2     |
| Learning Rate      | 0.01  |
| Weight Decay       | 0.001 |

```bash
Best validation MAE: 0.7510
```

Then, we train the neural network for 10 epochs using the best hyperparameters found during the random search, and evaluate its performance on the **test set**. The results are as follows:

```bash
NN test MAE: 0.7700
```

We see that the neural network performs better than all other methods.

To predict ratings for new user-movie pairs, we first extend the user embeddings for new users and set its value to the average of the trained embeddings. Then, we use the given ratings in the file to adapt the new user's embedding via gradient descent (lr=0.01, num_steps=50, wd=0.0). We then use the trained and adapted neural network to make predictions for the unrated user-movie pairs in the file.

The corresponding results from the **online submission system** are as follows:

| Method | MAE of GIVEN 5 | MAE of GIVEN 10 | MAE of GIVEN 20 | OVERALL MAE |
| ------ | -------------- | --------------- | --------------- | ----------- |
| NN     | 0.8144         | 0.7593          | 0.7342          | 0.7667      |
