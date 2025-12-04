import chess
import torch
import numpy as np
import time
from .mcts import MCTS
from .chess_utils import encode_board
from .model.model_utils import get_device

def self_play_game(model, num_simulations=100, temperature=1.0, temp_threshold=30):
    """
    Plays a single game of self-play.
    Returns a list of training examples: (state_tensor, policy_target, value_target)
    """
    board = chess.Board()
    history = [] # List of board states
    examples = [] # List of (state_tensor, policy, current_player)
    
    # Try to import C++ MCTS
    try:
        import chess_engine_cpp
        USE_CPP_MCTS = True
    except ImportError:
        USE_CPP_MCTS = False
        
    device = get_device()
    
    if USE_CPP_MCTS:
        # C++ MCTS takes (predict_fn, num_simulations, c_puct)
        # It expects predict_fn to return (p, v)
        # model passed here is 'predict_fn' from run_self_play_loop (ModelServer wrapper)
        # or a PyTorch model.
        
        # If it's a PyTorch model, we need to wrap it to accept numpy
        if isinstance(model, torch.nn.Module):
            real_model = model
            def predict_wrapper(state_array):
                tensor = torch.from_numpy(state_array).unsqueeze(0).to(device)
                with torch.no_grad():
                    p, v = real_model(tensor)
                return p.cpu().numpy()[0], v.item()
            mcts = chess_engine_cpp.MCTS(predict_wrapper, num_simulations, 1.0)
        else:
            # It's already a callable (ModelServer wrapper)
            mcts = chess_engine_cpp.MCTS(model, num_simulations, 1.0)
    else:
        mcts = MCTS(model, device, num_simulations=num_simulations)
    
    move_count = 0
    from .chess_utils import ACTION_CONVERTER
    
    while not board.is_game_over():
        # Temperature scheduling
        if move_count >= temp_threshold:
            temp = 0.05 # Almost deterministic
        else:
            temp = temperature
            
        # Run MCTS
        if USE_CPP_MCTS:
            # C++ search takes FEN
            best_move_uci, visits = mcts.search(board.fen())
            
            # Reconstruct policy
            policy = np.zeros(4672, dtype=np.float32)
            total_visits = 0
            for move_uci, count in visits:
                move = chess.Move.from_uci(move_uci)
                idx = ACTION_CONVERTER.encode(move, board.turn)
                if idx is not None:
                    policy[idx] = count
                    total_visits += count
            
            if total_visits > 0:
                policy /= total_visits
                
            value_est = 0.0 # Not returned by C++ currently
        else:
            policy, value_est = mcts.search(board, history)
        
        # Store example
        # We store the board state (encoded), the MCTS policy, and the current player
        from .model.nnue import encode_halfkp
        state_tensor = encode_board(board, history)
        sparse_features = encode_halfkp(board)
        examples.append([state_tensor, sparse_features, policy, board.turn])
        
        # Select move
        if temp == 0:
            action_idx = np.argmax(policy)
        else:
            # Sample from policy
            # Add small epsilon to avoid div by zero if policy is all zeros (shouldn't happen)
            policy = policy.astype(np.float64)
            policy /= np.sum(policy)
            action_idx = np.random.choice(len(policy), p=policy)
            
        # Decode action
        from .chess_utils import ACTION_CONVERTER
        move = ACTION_CONVERTER.decode(action_idx, board.turn)
        
        if move is None or move not in board.legal_moves:
            # Fallback (should not happen if logic is correct)
            print(f"WARNING: Selected illegal move index {action_idx}. Fallback to random.")
            move = np.random.choice(list(board.legal_moves))
            
        # Update history
        history.append(board.copy()) # Store copy of previous state
        if len(history) > 7:
            history.pop(0) # Keep last 7
            
        # Make move
        board.push(move)
        move_count += 1
        
        if move_count > 300:
            # Draw by adjudication (too long)
            break
            
    # Game Over
    result = board.result()
    if result == '1-0':
        winner = chess.WHITE
    elif result == '0-1':
        winner = chess.BLACK
    else:
        winner = None # Draw
        
    # Assign rewards
    processed_examples = []
    for state_tensor, sparse_features, policy, turn in examples:
        if winner is None:
            z = 0.0
        elif turn == winner:
            z = 1.0
        else:
            z = -1.0
            
        processed_examples.append((state_tensor, sparse_features, policy, z))
        
    return processed_examples

def run_self_play_loop(model, num_games=10, num_simulations=100, num_workers=4):
    """
    Runs a loop of self-play games in parallel.
    """
    from .model_server import ModelServer
    from .model.model_utils import get_device
    import concurrent.futures
    
    device = get_device()
    
    # Start Model Server
    # Note: If model is already on device, good.
    server = ModelServer(model, device, max_batch_size=num_workers)
    server.start()
    
    # Wrapper for MCTS
    def predict_fn(state_array):
        future = server.predict(state_array)
        return future.result()
        
    total_examples = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_games):
            # We submit games. If num_games > num_workers, they queue up.
            # We pass predict_fn as the 'model' to self_play_game
            # But self_play_game expects 'model' and passes it to MCTS.
            # MCTS now handles callable model.
            futures.append(executor.submit(self_play_game, predict_fn, num_simulations))
            
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                game_examples = future.result()
                total_examples.extend(game_examples)
                print(f"Game finished. Generated {len(game_examples)} examples. ({len(total_examples)} total)")
            except Exception as e:
                print(f"Game failed: {e}")
                import traceback
                traceback.print_exc()
                
    server.stop()
    
    end_time = time.time()
    print(f"Self-play finished in {end_time - start_time:.2f}s.")
    
    return total_examples
