# AppleSiliconChess: Deep Dive & Documentation

**AppleSiliconChess** is a high-performance, self-learning chess engine designed for Apple Silicon. It implements the **AlphaZero** algorithm with a hybrid Python/C++ architecture and **NNUE** distillation.

---

## 📚 Table of Contents
1.  [Mathematical Foundations](#-mathematical-foundations)
2.  [Codebase Deep Dive](#-codebase-deep-dive)
    *   [Training Loop (`src/train.py`)](#srctrainpy)
    *   [Self-Play Engine (`src/self_play.py`)](#srcself_playpy)
    *   [Monte Carlo Tree Search (`cpp/mcts.cpp`)](#cppmctscpp)
    *   [Neural Networks (`src/model/`)](#srcmodel)
3.  [Installation & Usage](#-installation--usage)

---

## 🧮 Mathematical Foundations

### 1. The AlphaZero Objective
The goal is to approximate the optimal policy $\pi^*(s)$ and value $v^*(s)$ for any board state $s$.
We use a neural network $f_\theta(s) = (\mathbf{p}, v)$ where:
*   $\mathbf{p}$: A vector of probabilities over all legal moves (the "prior").
*   $v$: A scalar in $[-1, 1]$ estimating the expected game outcome.

The network is trained to minimize the following loss function:
$$ L = (z - v)^2 - \boldsymbol{\pi}^T \log \mathbf{p} + c ||\theta||^2 $$
*   $(z - v)^2$: **Mean Squared Error**. $z$ is the actual game result (+1/-1). We want $v$ to predict $z$.
*   $-\boldsymbol{\pi}^T \log \mathbf{p}$: **Cross-Entropy Loss**. $\boldsymbol{\pi}$ is the "improved" policy from MCTS. We want the network's prior $\mathbf{p}$ to match the MCTS search probabilities.
*   $c ||\theta||^2$: **L2 Regularization** (Weight Decay) to prevent overfitting.

### 2. Monte Carlo Tree Search (MCTS)
MCTS is the engine's "reasoning" process. It builds a search tree to improve upon the raw network predictions.

**The PUCT Formula (Selection Phase)**
At each node, we select the child action $a$ that maximizes:
$$ U(s, a) = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} $$
*   **$Q(s, a)$**: The mean value of action $a$ (Exploitation).
*   **$P(s, a)$**: The prior probability from the network (Guidance).
*   **$\frac{\sqrt{\sum N}}{1+N}$**: The exploration term. As we visit other nodes more, this term grows for unvisited nodes, encouraging exploration.

**Backpropagation**
When we reach a leaf and evaluate it as $v$, we update the path:
$$ Q(s, a) = \frac{N(s, a) \cdot Q(s, a) + v}{N(s, a) + 1} $$
$$ N(s, a) \leftarrow N(s, a) + 1 $$

### 3. NNUE Distillation
To speed up the engine, we distill the heavy ResNet (Teacher) into a fast NNUE (Student).
$$ L_{distill} = \alpha \cdot \text{MSE}(V_{student}, V_{teacher}) + \beta \cdot \text{KL}(P_{student} || P_{teacher}) $$
This forces the Student to mimic the Teacher's outputs, effectively compressing the knowledge.

---

## 🔍 Codebase Deep Dive

### `src/train.py`
**Purpose**: The central command center. It orchestrates the cycle of data generation, training, and validation.

*   **`main()`**:
    *   **Initialization**: Sets up `ChessResNet` (Teacher) and `ReplayBuffer`.
    *   **Loop (`NUM_ITERATIONS`)**:
        1.  **`run_self_play_loop`**: Generates games using the current best model.
        2.  **`Trainer.train_epoch`**: Trains a *candidate* model on the replay buffer.
        3.  **`run_arena`**: The **Gatekeeper**. Matches Candidate vs Best.
            *   If `win_rate > 0.55`: Candidate becomes Best.
            *   Else: Candidate is discarded.

### `src/self_play.py`
**Purpose**: Generates training data by playing games against itself.

*   **`run_self_play_loop(model, ...)`**:
    *   Starts a `ModelServer` (see below) to handle GPU batching.
    *   Uses `ThreadPoolExecutor` to launch `num_workers` (e.g., 8) parallel game threads.
    *   Collects examples from all threads.

*   **`self_play_game(predict_fn, ...)`**:
    *   **Logic**: Plays one game from start to finish.
    *   **`mcts.search(board)`**: Calls the C++ MCTS to get the best move and policy $\boldsymbol{\pi}$.
    *   **Data Storage**: Saves `(state, policy, value)` to a list.
    *   **Temperature**: For the first 30 moves, picks moves proportionally to visit counts (exploration). After 30, picks the best move (exploitation).
    *   **Result**: When game ends, assigns $z$ (+1/-1) to all stored positions.

### `src/model_server.py`
**Purpose**: Optimizes GPU throughput.
*   **Problem**: 8 CPU threads sending 1 item each to GPU is slow.
*   **Solution**: `ModelServer` sits in a loop.
    *   **`predict(state)`**: Threads call this. It pushes the state to a `Queue` and returns a `Future`.
    *   **`_loop`**: Pulls items from Queue, stacks them into a Batch (e.g., 64 items), runs `model(batch)`, and completes the Futures.

### `cpp/mcts.cpp` (The Engine Core)
**Purpose**: High-performance tree search. Rewritten in C++ to bypass Python overhead.

*   **`Node` struct**:
    *   Stores `visit_count`, `value_sum`, `prior`, and a map of `children`.
*   **`MCTS::search(fen)`**:
    *   **Root**: Creates a root node from the FEN.
    *   **Simulations Loop**:
        1.  **Selection (`_select_child`)**: Traverses down using the PUCT formula.
        2.  **Expansion (`_expand`)**: Generates legal moves using `chess-library`.
        3.  **Evaluation (`_expand_and_evaluate`)**:
            *   Calls back to Python: `model(encode_board(state))`.
            *   Gets `(policy, value)`.
        4.  **Backpropagation**: Updates $Q$ and $N$ up the tree.
    *   **Return**: Returns the best move and the visit counts (policy).

### `src/model/resnet.py`
**Purpose**: The "Brain" (Teacher).
*   **`ChessResNet`**:
    *   **Input**: (Batch, 119, 8, 8). 119 planes representing pieces, history, etc.
    *   **`ResBlock`**: `Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add(Input) -> ReLU`.
    *   **Heads**:
        *   `PolicyHead`: Conv1x1 -> FC -> Softmax (4672 outputs).
        *   `ValueHead`: Conv1x1 -> FC -> Tanh (1 output).

### `src/chess_utils.py`
**Purpose**: Data encoding.
*   **`encode_board`**: Converts a `chess.Board` into the $119 \times 8 \times 8$ tensor.
    *   Planes 0-11: My pieces (P, N, B, R, Q, K).
    *   Planes 12-23: Opponent pieces.
    *   Planes 24-110: History (previous 7 positions).
    *   Planes 111-118: Metadata (Color, Castling Rights, Move Counts).
*   **`ActionConverter`**: Maps 4,672 policy indices to UCI moves (e.g., `e2e4` -> `1234`).
    *   Encodes Queen moves (56 dirs), Knight moves (8), Underpromotions (9).

---

## 🛠️ Installation & Usage

1.  **Install**:
    ```bash
    pip install -r requirements.txt
    python3 setup.py build_ext --inplace
    ```
2.  **Train**:
    ```bash
    python3 -m src.train
    ```
3.  **Play**:
    ```bash
    python3 -m src.main
    ```
