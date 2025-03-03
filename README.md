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

First, we randomly split the data into training, validation, and test sets of sizes **90%, 5%, and 5%** respectively.

The scikit-learn implementations of the random forest and histogram gradient boosting algorithms are used.

As for the neural network, we use a custom PyTorch implementation. The model is based on the idea that users and movies can be represented by embeddings of arbitary size, and that we can use gradient descent to find the optimal embedding vectors. Simple approaches include computing a dot product between the user and movie embeddings, or even doing so with a learned bias for each user and movie, and adjusting the embedding vectors using gradient descent. However, the neural network presented here extends this idea by concatenating the user and movie embeddings and passing them through a **multilayer perceptron** (where embeddings and linear layers are initialized with values sampled from a normal distribution with mean 0 and standard deviation 0.01, and biases are initialized to 0) so that the model may learn more complex representations of users, movies, and interactions between them. Regardless of the approach, a **sigmoid activation function** is used for the output layer to ensure that the predicted ratings are bounded between 1 and 5.

Then, we assess the performance of the user and item-based algorithms in addition to various machine learning models (the neural network (NN) here is untrained) on the **test set**. The results are as follows:

| Method                    | MAE    |
| ------------------------- | ------ |
| Cosine Similarity         | 1.0195 |
| Pearson                   | 1.2200 |
| Pearson IUF               | 1.0587 |
| Pearson IUF with Case Mod | 1.0657 |
| Item-based                | 1.0698 |
| Random Forest             | 0.8341 |
| Hist Gradient Boosting    | 0.8280 |
| NN                        | 1.0462 |

The baseline results show that traditional methods using a nearest neighbors approach, although simple to implement, have relatively high MAE due to limitations in the ability otocapture complex interactions. The machine learning models achieve lower MAE because they can capture more complex relationships than just cosine similarity or pearson correlation. However, they can still be limited in their ability to capture complex interactions due to the geometry of their decision boundaries. The untrained Neural Network has a similar MAE to the traditional methods, which is expected since the model has yet to be trained. When trained, it can learn even more complex representations and interactions, which may lead to improved performance. However, it is computational intensive to train and its performance is sensitive to hyperparameter choices.

To train the neural network, we first run **random search for 50 iterations** to find appropriate hyperparameters. For each iteration, the model is trained for **10 epochs**, and its performance on the **validation set** determines whether it is the best model so far. The best set of hyperparameters is as follows:

| Hyperparameter     | Value  |
| ------------------ | ------ |
| Batch Size         | 128    |
| Embed Dropout Rate | 0.1    |
| Dropout Rate       | 0.3    |
| Act Max            | 8192   |
| Hidden Layers      | 4      |
| Learning Rate      | 0.0001 |
| Weight Decay       | 0.01   |

```bash
Best validation MAE: 0.7651
```

We see that the neural network performs better than the other methods.

Then, we train the neural network for 10 epochs using the best hyperparameters found during the random search, and evaluate its performance on the **test set**. The results are as follows:

```bash
NN test MAE: 0.7662
```

To predict ratings for new user-movie pairs, we first extend the user embeddings for new users. Then, for each new user, we use **only** the trained embeddings and the 5 given ratings in file to train the neural network (for 100 steps) to learn the optimal embedding vectors for the new user. To ensure better performance and stable predictions when given only 5 ratings, we halve the learning rate and double the number of steps, and afterwards blend the adapted embedding with the global average embedding using a factor of 0.5.

We then use the trained and adapted neural network to make predictions for the unrated user-movie pairs in the file.

The corresponding results from the **online submission system** are as follows:

| Method | MAE of GIVEN 5 | MAE of GIVEN 10 | MAE of GIVEN 20 | OVERALL MAE |
| ------ | -------------- | --------------- | --------------- | ----------- |
| NN     | 0.8267         | 0.7620          | 0.7507          | 0.7784      |
