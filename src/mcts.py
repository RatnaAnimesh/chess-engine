import math
import numpy as np
import torch
import chess
from .chess_utils import encode_board, ACTION_CONVERTER

class MCTSNode:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {} # {move: MCTSNode}
        self.state = None # chess.Board (only for leaf nodes usually, but we might store it)
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, device, num_simulations=800, c_puct=1.0, batch_size=8, tablebase=None):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.tablebase = tablebase

    def _evaluate(self, state_array):
        """
        Evaluates the state.
        state_array: (119, 8, 8) numpy array
        Returns: (policy_logits, value)
        """
        # Check if model is a callable (function) or a Module
        if isinstance(self.model, torch.nn.Module):
            tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                p, v = self.model(tensor)
            return p, v
        else:
            # Assume it's a callable that accepts numpy and returns (p, v)
            # This is for ModelServer
            return self.model(state_array)

    def search(self, root_board: chess.Board, history: list = None):
        """
        Performs MCTS search from the root position.
        Returns the move probabilities (policy) and the estimated value.
        """
        root = MCTSNode(prior=0.0)
        root.state = root_board.copy()
        
        # Expand root immediately
        self._expand_node(root, history)
        
        # Add Dirichlet noise to root priors (Exploration)
        self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_history = list(history) if history else []
            
            # 1. Selection
            while node.is_expanded:
                action, node = self._select_child(node)
                search_path.append(node)
                # Update history for the virtual traversal
                # Note: We don't actually need the full board state for intermediate nodes 
                # if we just want to reach a leaf. But to expand the leaf, we need the state.
                # So we must maintain the state in the nodes or re-play moves.
                # Re-playing is slower but saves memory. Storing state is faster.
                # Given Python, let's store state in nodes for now.
            
            leaf = node
            parent = search_path[-2] if len(search_path) > 1 else None
            
            # Check if game over
            if leaf.state.is_game_over():
                # Terminal state
                # Value is from the perspective of the player who just moved (parent's turn)
                # If leaf state is checkmate, then the side to move (leaf.state.turn) has lost.
                # So the value for the side to move is -1.
                # But we propagate value for the previous player.
                
                # Let's standardize: Value is always for the player at the root?
                # No, standard MCTS propagates value relative to the player at that node.
                # If leaf.state.turn is White, and White is mated, value is -1.
                # This value is backed up to the parent (Black's turn).
                # Black sees this as +1.
                
                result = leaf.state.result()
                if result == '1-0':
                    value = 1.0 if leaf.state.turn == chess.WHITE else -1.0
                elif result == '0-1':
                    value = 1.0 if leaf.state.turn == chess.BLACK else -1.0
                else:
                    value = 0.0
                
                self._backpropagate(search_path, value, leaf.state.turn)
                continue

            # 2. Expansion and Evaluation
            # We need to evaluate the leaf.
            # For efficiency, we should batch this. But for a simple implementation, we do one by one.
            # To support batching, we would collect leaves from multiple threads/coroutines.
            # Here we do synchronous for simplicity first.
            
            value = self._expand_and_evaluate(leaf, current_history)
            
            # 3. Backup
            self._backpropagate(search_path, value, leaf.state.turn)

        # Calculate output policy
        counts = {move: child.visit_count for move, child in root.children.items()}
        total_counts = sum(counts.values())
        
        from .chess_utils import ACTION_SIZE
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        
        # If we have no visits (shouldn't happen), return uniform
        if total_counts == 0:
            return policy, 0.0
            
        for move, count in counts.items():
            idx = ACTION_CONVERTER.encode(move, root_board.turn)
            if idx is not None:
                policy[idx] = count / total_counts
                
        return policy, root.value()

    def _select_child(self, node):
        # PUCT
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # We iterate over all legal moves (children)
        for move, child in node.children.items():
            # Q value (exploitation)
            # If never visited, Q is 0? Or we can use FPU (First Play Urgency).
            # AlphaZero uses Q=0 for unvisited.
            q_value = child.value()
            
            # U value (exploration)
            # U = c_puct * P * sqrt(N_parent) / (1 + N_child)
            u_value = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            
            # From the perspective of the current player (node.state.turn)
            # The child's value is from the perspective of the NEXT player.
            # So we must negate Q.
            # Q(s, a) should be the expected value for player 's'.
            # child.value() is the average value of the child node state.
            # If child node state is good for the next player, it's bad for us.
            score = -q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = move
                best_child = child
                
        return best_action, best_child

    def _expand_node(self, node, history):
        # Check Tablebase
        if self.tablebase is not None:
            # Count pieces
            if len(node.state.piece_map()) <= 7:
                try:
                    wdl = self.tablebase.get_wdl(node.state)
                    # wdl: 2 (Win), 1 (Win), 0 (Draw), -1 (Loss), -2 (Loss)
                    # We map to value [-1, 1]
                    if wdl > 0:
                        value = 1.0
                    elif wdl < 0:
                        value = -1.0
                    else:
                        value = 0.0
                        
                    # If TB hit, we don't need policy? 
                    # Actually, we still need policy to pick the BEST winning move.
                    # But for value, we use exact value.
                    # AlphaZero uses TB value but still uses network policy?
                    # Or uses TB DTZ (Distance To Zero) to pick move.
                    # For simplicity, we just override value and let network guide policy, 
                    # or we can just return value and not expand?
                    # If we don't expand, we can't select children.
                    # So we MUST expand. We just use TB value instead of Network Value.
                    
                    # We still run network for Policy.
                    pass
                except Exception:
                    # TB probe failed (maybe not in TB)
                    pass
        
        # Get legal moves
        legal_moves = list(node.state.legal_moves)
        
        # Evaluate network
        # Prepare input
        tensor = encode_board(node.state, history) # (119, 8, 8)
        
        if hasattr(self.model, 'predict_batch'):
            # It's likely our ModelServer wrapper or similar
            # But wait, MCTS is single threaded here.
            pass
            
        # We define a standard interface: self.forward_fn(tensor)
        # If self.model is a torch.nn.Module, we wrap it.
        
        policy_logits, value_net = self._evaluate(tensor)
            
        # Use TB value if available
        if self.tablebase is not None and len(node.state.piece_map()) <= 7:
            try:
                wdl = self.tablebase.get_wdl(node.state)
                if wdl is not None:
                    if wdl > 0: value = 1.0
                    elif wdl < 0: value = -1.0
                    else: value = 0.0
                else:
                    value = value_net
            except:
                value = value_net
        else:
            value = value_net
        
        # Policy
        # If _evaluate returns numpy, we don't need .cpu().numpy()
        # If it returns tensor, we do.
        if isinstance(policy_logits, torch.Tensor):
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        else:
            # Assume it's already logits or probs?
            # The model returns logits.
            # We need softmax.
            # If numpy:
            import numpy as np
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            policy_probs = softmax(policy_logits)
            
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        # Create children
        # We only create children for legal moves
        # We normalize the priors over legal moves
        policy_sum = 0
        
        for move in legal_moves:
            idx = ACTION_CONVERTER.encode(move, node.state.turn)
            if idx is not None:
                prior = policy_probs[idx]
                child = MCTSNode(prior)
                child.state = node.state.copy()
                child.state.push(move)
                node.children[move] = child
                policy_sum += prior
                
        # Re-normalize priors
        if policy_sum > 0:
            for child in node.children.values():
                child.prior /= policy_sum
                
        node.is_expanded = True
        return value

    def _expand_and_evaluate(self, node, history):
        return self._expand_node(node, history)

    def _backpropagate(self, search_path, value, turn):
        # value is the evaluation of the leaf node state (from leaf.turn perspective)
        # We propagate up.
        # If node.turn == turn, we add value.
        # If node.turn != turn, we subtract value (or add -value).
        
        # Actually, let's simplify.
        # Value v is from the perspective of the player who just moved to reach the leaf.
        # No, the network outputs v for the player whose turn it is in the state.
        # So v is for leaf.turn.
        
        for node in reversed(search_path):
            node.visit_count += 1
            
            # If this node's turn is the same as the leaf's turn, the value is positive for this node.
            # If different, it's negative.
            if node.state.turn == turn:
                node.value_sum += value
            else:
                node.value_sum -= value

    def _add_dirichlet_noise(self, node):
        epsilon = 0.25
        alpha = 0.3
        moves = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))
        
        for i, move in enumerate(moves):
            node.children[move].prior = (1 - epsilon) * node.children[move].prior + epsilon * noise[i]
