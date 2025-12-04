# AppleSiliconChess: A First-Principles Explanation

This document explains how the chess engine works, assuming no prior knowledge of Machine Learning (ML). We will build up the concepts from scratch, defining every term and the mathematics behind it.

---

## 1. The Core Problem: How do computers play chess?

At any point in a chess game, you have a **State** (the board position). You need to choose an **Action** (a move) that maximizes your chance of winning.

### The Search Tree
Imagine listing every possible move you can make. Then, for each of those moves, list every reply your opponent can make. Then every reply you can make to that... and so on. This structure is called a **Game Tree**.

*   **Root**: The current position.
*   **Branches**: The moves.
*   **Leaves**: The positions at the end of your lookahead.

The problem is that chess is too big. The number of possible games is estimated to be $10^{120}$ (more than atoms in the universe). You cannot search to the end. You must stop searching at some point and **evaluate** the position.

### The Evaluation Function
Traditionally (like in Stockfish), humans wrote code to score a position:
*   Pawn = 1 point, Knight = 3 points.
*   "Control the center" = +0.5 points.
*   "King safety" = +1.0 point.

**Our Approach (AlphaZero)**: We don't write these rules. We train a **Neural Network** to look at the board and tell us two things:
1.  **Policy ($p$)**: "Which moves look good immediately?" (Intuition).
2.  **Value ($v$)**: "Who is winning this position?" (Judgment).

---

## 2. The Neural Network (The "Brain")

A Neural Network is a mathematical function that takes numbers as input and produces numbers as output. It has internal parameters called **Weights** ($\theta$) that we can adjust.

### Input: The Tensor
Computers don't see "Rooks" or "Bishops". They see numbers. We convert the board into a 3D grid of numbers called a **Tensor**.
*   Dimensions: $119 \times 8 \times 8$.
*   Think of it as 119 layers of $8 \times 8$ chessboards stacked on top of each other.
    *   Layer 0: 1s where white pawns are, 0s elsewhere.
    *   Layer 1: 1s where white knights are...
    *   ...
    *   Layer 118: 1s if it is White's turn to move.

### The Architecture: ResNet (Residual Network)
We pass this tensor through a **Convolutional Neural Network (CNN)**.
*   **Convolution**: Imagine a small $3 \times 3$ window sliding over the board. It looks for patterns (e.g., "a pawn protecting another pawn").
*   **Layers**: We stack many of these. The first layer sees simple patterns (pawns). The next layer sees patterns of patterns (pawn chains). The final layer sees high-level concepts (fortresses, attacks).
*   **Residual Connection**: As networks get deep, they get hard to train. A "Residual Block" adds the input of a layer to its output: $y = f(x) + x$. This allows information to flow easily through the network.

### Output
The network outputs two things:
1.  **Policy ($\mathbf{p}$)**: A list of 4,672 numbers (one for every possible move in chess). These are probabilities (they sum to 1). A high number means "I think this move is good".
2.  **Value ($v$)**: A single number between -1 (I am losing) and +1 (I am winning).

---

## 3. Monte Carlo Tree Search (The "Reasoning")

The network gives us a "gut feeling". But gut feelings can be wrong. We need to verify them with calculation. This is **Monte Carlo Tree Search (MCTS)**.

Instead of searching every move equally (which is wasteful), MCTS focuses on moves that the Network thinks are good ($\mathbf{p}$) and that have yielded good results so far ($Q$).

### The PUCT Formula
At every step in the search tree, we choose the move $a$ that maximizes this formula:

$$ U(s, a) = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum N}}{1 + N(s, a)} $$

Let's break this down:
1.  **$Q(s, a)$ (Exploitation)**: The average value of this move so far. "This move worked well in previous simulations, so I should try it again."
2.  **$P(s, a)$ (Guidance)**: The probability from the Neural Network. "The Brain says this move is promising, so I should try it."
3.  **$\frac{\sqrt{\sum N}}{1 + N(s, a)}$ (Exploration)**:
    *   $N(s, a)$ is how many times we've tried move $a$.
    *   $\sum N$ is how many times we've visited the parent position.
    *   If we haven't tried move $a$ much (small $N$), this term becomes huge. It forces the engine to say: "Wait, I haven't checked this move yet. I should at least look at it."

### The Loop
1.  **Select**: Follow the PUCT formula down the tree until you hit a new position.
2.  **Evaluate**: Ask the Neural Network for its opinion ($\mathbf{p}, v$) on this new position.
3.  **Backup**: Propagate the value $v$ back up the tree. Update the average $Q$ for all moves we took.

After 800 simulations, we stop and pick the move we visited the most. This is our final move.

---

## 4. Reinforcement Learning (The "Training")

How do we find the right Weights ($\theta$) for the network? We let it play against itself.

### Self-Play
The engine plays thousands of games against itself.
*   In position $s_1$, MCTS calculates the best move probabilities $\boldsymbol{\pi}$.
*   It plays a move, reaches $s_2$, calculates $\boldsymbol{\pi}$, plays...
*   Eventually, the game ends. Let's say White wins ($z = +1$).

Now we have a dataset: "In position $s_1$, the MCTS said $\boldsymbol{\pi}$ was the best strategy, and the result was $+1$."

### The Loss Function
We train the network to predict these results. We define a **Loss Function** ($L$) that measures how "wrong" the network is.

$$ L = (z - v)^2 - \boldsymbol{\pi}^T \log \mathbf{p} $$

1.  **$(z - v)^2$ (Value Loss)**:
    *   $z$ is the truth (+1). $v$ is what the network predicted (e.g., 0.2).
    *   We want $(1 - 0.2)^2$ to be small. The network learns to predict who will win.
2.  **$-\boldsymbol{\pi}^T \log \mathbf{p}$ (Policy Loss)**:
    *   $\boldsymbol{\pi}$ is what the MCTS calculated (the "verified" truth).
    *   $\mathbf{p}$ is what the network predicted (the "gut feeling").
    *   We want the gut feeling ($\mathbf{p}$) to match the calculation ($\boldsymbol{\pi}$). This makes the network smarter, so the *next* MCTS search starts with better intuition.

We use **Backpropagation** (Calculus chain rule) to adjust the weights $\theta$ to reduce this Loss.

---

## 5. Distillation (The "Optimization")

The ResNet is smart but slow. It takes time to calculate.
For the final engine, we want speed.

We create a second, smaller network called **NNUE** (Efficiently Updatable Neural Network).
*   It's much simpler (just 2 layers).
*   It runs on the CPU, not the GPU.

We teach the NNUE to copy the ResNet.
*   Teacher (ResNet): "I think this position is +0.8."
*   Student (NNUE): "I think it's +0.5."
*   We adjust the Student to match the Teacher.

Because the Student is simple, it can run 100x faster. It might be slightly less smart than the Teacher, but because it can search 100x more positions in the same time, it ends up playing better chess.

---

## Summary
1.  **ResNet**: A deep brain that learns chess strategy from scratch.
2.  **MCTS**: A search algorithm that uses the brain to look ahead and verify moves.
3.  **Self-Play**: The engine plays itself to generate data.
4.  **Training**: The brain updates itself to predict the results of the self-play.
5.  **Distillation**: We compress the brain into a tiny, fast network (NNUE) for the final engine.
