import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.resnet import ChessResNet
from src.model.model_utils import get_device
from src.training.dataset import ReplayBuffer, ChessDataset
from src.training.trainer import Trainer
from src.self_play import run_self_play_loop

def main():
    # Configuration
    # Configuration
    NUM_ITERATIONS = 100 # Run for a long time
    NUM_SELF_PLAY_GAMES = 50 # Games per iteration
    BATCH_SIZE = 64
    EPOCHS = 5
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize Model
    # Initialize Model
    best_model = ChessResNet(num_blocks=4, num_filters=64)
    best_model.to(device)
    
    # Save initial best model
    torch.save(best_model.state_dict(), "best_model.pt")
    
    replay_buffer = ReplayBuffer(capacity=20000)
    
    for iteration in range(NUM_ITERATIONS):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        # 1. Self-Play with Best Model
        print("--- Self-Play Phase (Best Model) ---")
        # Load best model for self-play
        best_model.load_state_dict(torch.load("best_model.pt", map_location=device))
        
        new_examples = run_self_play_loop(best_model, num_games=NUM_SELF_PLAY_GAMES, num_simulations=400, num_workers=8)
        replay_buffer.extend(new_examples)
        print(f"Replay Buffer Size: {len(replay_buffer)}")
        
        # 2. Training Candidate
        print("--- Training Phase ---")
        # Create candidate as copy of best
        candidate_model = ChessResNet(num_blocks=4, num_filters=64).to(device)
        candidate_model.load_state_dict(best_model.state_dict())
        
        trainer = Trainer(candidate_model, device)
        dataset = ChessDataset(list(replay_buffer.buffer))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        for epoch in range(EPOCHS):
            metrics = trainer.train_epoch(dataloader)
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {metrics['loss']:.4f}")
            
        # 3. Arena (Gatekeeper)
        print("--- Arena Phase ---")
        # Play Candidate vs Best
        # We need a function to play matches.
        # We can reuse run_self_play_loop logic but with two models?
        # Or just write a simple match loop here.
        
        wins, losses, draws = run_arena(candidate_model, best_model, num_games=20, simulations=100)
        print(f"Arena Results - Wins: {wins}, Losses: {losses}, Draws: {draws}")
        
        total_games = wins + losses + draws
        win_rate = (wins + 0.5 * draws) / total_games
        
        if win_rate >= 0.55:
            print(f"Candidate Accepted! (Win Rate: {win_rate:.2f})")
            torch.save(candidate_model.state_dict(), "best_model.pt")
            torch.save(candidate_model.state_dict(), f"checkpoint_{iteration}.pt")
        else:
            print(f"Candidate Rejected. (Win Rate: {win_rate:.2f})")
            # We discard candidate, next iteration uses old best_model

def run_arena(model1, model2, num_games=20, simulations=100):
    """
    Plays games between model1 and model2.
    Returns (wins, losses, draws) for model1.
    """
    import chess
    from src.self_play import self_play_game
    # We need a specialized function for 1v1.
    # self_play_game uses one model for both sides.
    # We can modify self_play_game or write a new one.
    # Writing a simple one here for clarity.
    
    # Actually, to use C++ MCTS efficiently, we should use the same infrastructure.
    # But C++ MCTS is designed for self-play (one model).
    # To play 1v1, we need to swap models based on turn.
    
    # Simplified Python loop for Arena (slower but functional)
    # Or we can use the C++ MCTS if we instantiate two of them.
    
    from src.mcts import MCTS
    # We will use Python MCTS for Arena to avoid complexity of swapping C++ backends dynamically
    # unless we expose that in C++.
    # Let's use the Python MCTS for Arena for now, or the C++ one if we can.
    
    # Actually, we can just instantiate two MCTS objects.
    # If using C++, we create two chess_engine_cpp.MCTS objects.
    
    try:
        import chess_engine_cpp
        USE_CPP = True
    except:
        USE_CPP = False
        
    device = get_device()
    
    # Prepare wrappers
    def get_mcts(model):
        if USE_CPP:
            if isinstance(model, torch.nn.Module):
                def predict_wrapper(state_array):
                    tensor = torch.from_numpy(state_array).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p, v = model(tensor)
                    return p.cpu().numpy()[0], v.item()
                return chess_engine_cpp.MCTS(predict_wrapper, simulations, 1.0)
            else:
                return chess_engine_cpp.MCTS(model, simulations, 1.0)
        else:
            return MCTS(model, device, num_simulations=simulations)

    mcts1 = get_mcts(model1)
    mcts2 = get_mcts(model2)
    
    wins = 0
    losses = 0
    draws = 0
    
    for i in range(num_games):
        # Swap colors every game
        # Game i: P1=Model1 (White), P2=Model2 (Black) if i even
        # Game i: P1=Model2 (White), P2=Model1 (Black) if i odd
        
        board = chess.Board()
        
        # If i is even, model1 is White.
        white_player = mcts1 if i % 2 == 0 else mcts2
        black_player = mcts2 if i % 2 == 0 else mcts1
        
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                mcts = white_player
            else:
                mcts = black_player
                
            if USE_CPP:
                best_move_uci, _ = mcts.search(board.fen())
                move = chess.Move.from_uci(best_move_uci)
            else:
                policy, _ = mcts.search(board, [])
                # Greedy in Arena
                idx = np.argmax(policy)
                from src.chess_utils import ACTION_CONVERTER
                move = ACTION_CONVERTER.decode(idx, board.turn)
                if move not in board.legal_moves:
                     move = list(board.legal_moves)[0] # Fallback
            
            board.push(move)
            if board.fullmove_number > 150:
                break # Draw
                
        res = board.result()
        if res == '1-0':
            if i % 2 == 0: wins += 1
            else: losses += 1
        elif res == '0-1':
            if i % 2 == 0: losses += 1
            else: wins += 1
        else:
            draws += 1
            
        print(f"Arena Game {i+1}/{num_games}: {res}")
        
    return wins, losses, draws

if __name__ == "__main__":
    main()
