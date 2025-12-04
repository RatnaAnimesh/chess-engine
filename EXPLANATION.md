# AppleSiliconChess: A Rigorous Technical Analysis

**Abstract**
This document presents a comprehensive analysis of the *AppleSiliconChess* engine, a reinforcement learning system based on the AlphaZero framework. It is intended for readers with no prior background in machine learning. We will derive the system's functionality from first principles, covering the necessary linear algebra, calculus, and probability theory required to understand the architecture, the search algorithm (Monte Carlo Tree Search), and the optimization process (Stochastic Gradient Descent).

---

## 1. The Mathematical Representation of Chess

To apply mathematical operations to a game of chess, we must first map the discrete game state into a continuous vector space.

### 1.1 The State Space as a Tensor
A chess board is an $8 \times 8$ grid. A standard representation involves mapping the presence of pieces to binary values.
Let $S$ be the set of all possible legal board states. We define a function $\phi: S \to \mathbb{R}^{119 \times 8 \times 8}$ that maps a state $s$ to a tensor (a multi-dimensional array) $X$.

The tensor $X$ consists of 119 "planes" (matrices of size $8 \times 8$):
*   **Planes 1-12**: Represent the location of each piece type (Pawn, Knight, Bishop, Rook, Queen, King) for both White and Black. $X_{c, i, j} = 1$ if a piece of type $c$ is at rank $i$, file $j$, and $0$ otherwise.
*   **Planes 13-119**: Encode history (previous board states to detect repetitions) and metadata (castling rights, turn indicator).

This transformation allows us to treat the board state as a point in a high-dimensional Euclidean space, upon which we can perform differentiable operations.

---

## 2. Neural Networks: Function Approximation

The core of the engine is a function $f_\theta: \mathbb{R}^{n} \to \mathbb{R}^{m}$ that approximates the optimal evaluation of a position. This function is parameterized by a set of weights $\theta$.

### 2.1 The Neuron (Perceptron)
The fundamental unit is the neuron. It performs an affine transformation followed by a non-linear activation.
Given an input vector $\mathbf{x} \in \mathbb{R}^d$, a neuron computes:
$$ y = \sigma(\mathbf{w} \cdot \mathbf{x} + b) $$
*   $\mathbf{w} \in \mathbb{R}^d$: The **weight vector**. It determines the importance of each input.
*   $b \in \mathbb{R}$: The **bias**. It shifts the activation threshold.
*   $\cdot$: The dot product $\sum w_i x_i$.
*   $\sigma$: A non-linear **activation function**. We use the Rectified Linear Unit (ReLU):
    $$ \text{ReLU}(z) = \max(0, z) $$
    This non-linearity is crucial. Without it, a stack of layers would collapse into a single linear transformation (since the composition of linear functions is linear).

### 2.2 Convolutional Neural Networks (CNNs)
Chess has **translation invariance**: a pawn structure on the left side of the board behaves similarly to one on the right. A fully connected network (where every input connects to every output) ignores this spatial structure.

We use **Convolution**. A "kernel" $K$ (a small $3 \times 3$ matrix of weights) slides over the input matrix $I$. The output $O$ at position $(i, j)$ is the sum of element-wise products:
$$ O_{i,j} = (I * K)_{i,j} = \sum_{m=0}^{2} \sum_{n=0}^{2} I_{i+m, j+n} K_{m,n} $$
This operation detects local features (e.g., a piece protecting another) regardless of their position on the board.

### 2.3 Residual Blocks (ResNet)
As we stack many convolutional layers to detect complex patterns, gradients (the signals used for learning) tend to vanish (become zero) during backpropagation, making training impossible.
A **Residual Block** solves this by adding a "skip connection":
$$ \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x} $$
where $\mathcal{F}$ is the stack of convolutional layers. The addition of $\mathbf{x}$ ensures that the gradient can flow directly through the network, allowing us to train very deep networks (e.g., 20+ layers).

---

## 3. The Objective Function and Optimization

We want our network $f_\theta(s)$ to output two things:
1.  **Policy ($\mathbf{p}$)**: A probability distribution over legal moves. $\mathbf{p} \in [0, 1]^{4672}$, $\sum p_i = 1$.
2.  **Value ($v$)**: An estimate of the win probability. $v \in [-1, 1]$.

### 3.1 The Loss Function
We measure the error of our network using a **Loss Function** $L(\theta)$.
Let $\boldsymbol{\pi}$ be the "true" best move probabilities (derived from search) and $z$ be the actual game outcome ($+1$ for win, $-1$ for loss).

$$ L(\theta) = (z - v)^2 - \sum_{i} \pi_i \log p_i + c ||\theta||^2 $$

1.  **Mean Squared Error ($(z-v)^2$)**: Measures the distance between the predicted value $v$ and the actual result $z$. Minimizing this makes the network a better evaluator.
2.  **Cross-Entropy ($-\sum \pi \log p$)**: Measures the divergence between the target distribution $\boldsymbol{\pi}$ and the predicted distribution $\mathbf{p}$. Minimizing this maximizes the likelihood of the correct moves.
3.  **L2 Regularization ($||\theta||^2$)**: Penalizes large weights. This prevents "overfitting" (memorizing specific positions instead of learning general rules).

### 3.2 Stochastic Gradient Descent (SGD)
To find the optimal weights $\theta^*$ that minimize $L(\theta)$, we use calculus.
The gradient $\nabla_\theta L$ is a vector of partial derivatives pointing in the direction of steepest ascent.
$$ \nabla_\theta L = \left[ \frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \dots \right] $$
We update the weights iteratively by moving in the opposite direction:
$$ \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t) $$
where $\eta$ is the **learning rate** (step size).

**Backpropagation**: To compute $\nabla_\theta L$ efficiently, we use the Chain Rule of calculus. We compute the error at the output and propagate it backward through the layers to find how much each weight contributed to the error.

---

## 4. Monte Carlo Tree Search (MCTS)

The neural network provides a heuristic (an approximation). To play perfectly, we need to search the game tree. However, the game tree is too large ($10^{120}$ nodes). MCTS allows us to search selectively.

### 4.1 The Multi-Armed Bandit Problem
At each node in the tree, we face a "Multi-Armed Bandit" problem: we have $k$ moves (arms), and we want to find the one with the highest expected reward. We must balance **Exploitation** (playing the move that looks best so far) and **Exploration** (trying moves we haven't tested enough).

### 4.2 The PUCT Algorithm
AlphaZero uses a variant of the Upper Confidence Bound (UCB) algorithm called PUCT. We select the child node $a$ that maximizes:
$$ a_t = \text{argmax}_a \left( Q(s, a) + U(s, a) \right) $$
$$ U(s, a) = c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} $$

*   **$Q(s, a)$**: The mean value of action $a$ observed so far.
*   **$P(s, a)$**: The prior probability from the neural network. This biases the search towards moves the network thinks are good.
*   **$\frac{\sqrt{\sum N}}{1 + N}$**: This term decreases as $N(s, a)$ increases. If a move is neglected, this term grows, eventually forcing the algorithm to visit it.

This guarantees that as the number of simulations $N \to \infty$, the search converges to the optimal minimax move.

---

## 5. Knowledge Distillation

For the final engine, we use a technique called **Distillation**.
We have a large, accurate model (Teacher, $T$) and a small, fast model (Student, $S$).
We train $S$ to minimize the Kullback-Leibler (KL) Divergence between its output and $T$'s output.

$$ D_{KL}(P_T || P_S) = \sum_{i} P_T(i) \log \frac{P_T(i)}{P_S(i)} $$

Minimizing this is equivalent to minimizing the Cross-Entropy between $T$ and $S$.
The Student learns to mimic the Teacher's probability distribution. Because the Student is smaller (NNUE architecture), it can be evaluated orders of magnitude faster, allowing for deeper searches in the same amount of time.

---

## Summary of the Algorithm

1.  **Initialization**: Randomly initialize parameters $\theta$.
2.  **Self-Play**:
    *   Run MCTS simulations using $f_\theta$ to guide the search.
    *   Generate a policy $\boldsymbol{\pi}$ from the visit counts $N(s, a)$.
    *   Play a move sampled from $\boldsymbol{\pi}$.
    *   Repeat until game ends with result $z$.
    *   Store $(s, \boldsymbol{\pi}, z)$ in a dataset $D$.
3.  **Training**:
    *   Sample a batch of positions from $D$.
    *   Compute the gradient $\nabla_\theta L$.
    *   Update $\theta \leftarrow \theta - \eta \nabla_\theta L$.
4.  **Iteration**: Repeat steps 2-3. The network improves the search, and the search generates better data for the network. This positive feedback loop leads to superhuman play.
