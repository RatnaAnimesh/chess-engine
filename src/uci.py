import sys
import chess
import torch
import threading
import time
import numpy as np
from .model.resnet import ChessResNet
from .model.model_utils import get_device
from .mcts import MCTS
from .chess_utils import ACTION_CONVERTER

class UCI:
    def __init__(self):
        self.board = chess.Board()
        self.device = get_device()
        # Load model (Teacher for now, or Student if available)
        # For this demo, we load a fresh Teacher. In prod, load checkpoint.
        self.model = ChessResNet(num_blocks=4, num_filters=64)
        self.model.to(self.device)
        self.model.eval()
        
        self.mcts = MCTS(self.model, self.device, num_simulations=800)
        self.searching = False
        self.stop_event = threading.Event()
        self.search_thread = None

    def loop(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                    
                self.handle_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Log error but don't crash
                # print(f"info string Error: {e}")
                pass

    def handle_command(self, line):
        parts = line.split()
        cmd = parts[0]
        
        if cmd == "uci":
            print("id name AppleSiliconChess")
            print("id author AnimeshRatna")
            print("option name SyzygyPath type string default <empty>")
            print("uciok")
            sys.stdout.flush()
            
        elif cmd == "setoption":
            # setoption name SyzygyPath value /path/to/tb
            if len(parts) >= 5 and parts[1] == "name" and parts[2] == "SyzygyPath" and parts[3] == "value":
                path = " ".join(parts[4:])
                try:
                    import chess.syzygy
                    self.tablebase = chess.syzygy.open_tablebase(path)
                    self.mcts.tablebase = self.tablebase
                    print(f"info string Syzygy tablebase loaded from {path}")
                except Exception as e:
                    print(f"info string Failed to load Syzygy: {e}")
            
        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()
            
        elif cmd == "ucinewgame":
            self.board = chess.Board()
            
        elif cmd == "position":
            self.handle_position(parts[1:])
            
        elif cmd == "go":
            self.handle_go(parts[1:])
            
        elif cmd == "stop":
            if self.searching:
                self.stop_event.set()
                if self.search_thread:
                    self.search_thread.join()
            
        elif cmd == "quit":
            sys.exit(0)

    def handle_position(self, args):
        # position [startpos | fen <fenstring>] [moves <move1> ... <movei>]
        if args[0] == "startpos":
            self.board = chess.Board()
            moves_idx = 1
        elif args[0] == "fen":
            # FEN can be multiple tokens
            # We need to find where 'moves' starts
            try:
                moves_idx = args.index("moves")
                fen = " ".join(args[1:moves_idx])
            except ValueError:
                fen = " ".join(args[1:])
                moves_idx = len(args)
            
            self.board = chess.Board(fen)
        else:
            return

        if moves_idx < len(args) and args[moves_idx] == "moves":
            for move_uci in args[moves_idx+1:]:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except:
                    pass

    def handle_go(self, args):
        # go wtime <x> btime <y> movestogo <z> ...
        # For simplicity, we just run fixed simulations or fixed time if specified
        # If infinite, we run until stop.
        
        # Parse limits (simplified)
        # We'll just launch a thread that runs MCTS.
        
        if self.searching:
            return
            
        self.stop_event.clear()
        self.searching = True
        
        # Run search in thread
        self.search_thread = threading.Thread(target=self.search)
        self.search_thread.start()

    def search(self):
        # We need to modify MCTS to respect stop_event
        # Since our MCTS.search is a loop over simulations, we can check stop_event there.
        # But MCTS class doesn't know about stop_event.
        # We will wrap the MCTS search or modify MCTS to accept a callback/event.
        
        # For now, let's just run the standard MCTS search.
        # If we want to support 'stop', we need to break the MCTS loop.
        # Let's assume we run a fixed number of simulations for this MVP.
        
        policy, value = self.mcts.search(self.board)
        
        # Pick best move
        # Argmax of policy
        # Filter legal moves
        best_move = None
        best_prob = -1
        
        legal_moves = list(self.board.legal_moves)
        
        # We need to map policy indices back to moves
        # This is slow if we iterate all 4672.
        # Better: iterate legal moves and check their prob.
        
        for move in legal_moves:
            idx = ACTION_CONVERTER.encode(move, self.board.turn)
            if idx is not None:
                prob = policy[idx]
                if prob > best_prob:
                    best_prob = prob
                    best_move = move
                    
        if best_move is None:
            # Fallback
            best_move = np.random.choice(legal_moves)
            
        print(f"bestmove {best_move.uci()}")
        sys.stdout.flush()
        
        self.searching = False
