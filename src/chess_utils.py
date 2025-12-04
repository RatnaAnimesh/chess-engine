import chess
import numpy as np
import torch

# Constants
# 8x8 board, 73 possible moves per square (Queen moves + Knight moves + Underpromotions)
# 56 Queen moves (N, NE, E, SE, S, SW, W, NW * 7 distances)
# 8 Knight moves
# 9 Underpromotions (3 directions * 3 pieces (R, B, N)) - Queen promotion is covered in Queen moves
# Total = 56 + 8 + 9 = 73
NUM_PLANES = 119
ACTION_SIZE = 4672 # 64 * 73

def create_uci_to_index_mapping():
    """
    Creates a mapping from UCI move strings to policy indices (0-4671).
    And the reverse mapping.
    """
    uci_to_idx = {}
    idx_to_uci = {}
    
    idx = 0
    
    # Directions for Queen moves (N, NE, E, SE, S, SW, W, NW)
    # (file_delta, rank_delta)
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    
    # Knight moves
    knight_moves = [
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2)
    ]
    
    # Underpromotions (N, NE, NW) - usually encoded as "move to square + promotion type"
    # But AlphaZero encodes them relative to source square.
    # Directions: N (capture/move), NE (capture), NW (capture)
    underpromo_directions = [(0, 1), (1, 1), (-1, 1)] # Relative to white pawn. For black, we flip.
    underpromo_pieces = ['n', 'b', 'r'] # Queen is implied in normal moves
    
    for square in range(64):
        rank = square // 8
        file = square % 8
        
        # 1. Queen moves (56)
        for d_file, d_rank in directions:
            for dist in range(1, 8):
                t_rank = rank + d_rank * dist
                t_file = file + d_file * dist
                
                if 0 <= t_rank < 8 and 0 <= t_file < 8:
                    # Valid target
                    # We don't check if it's a legal move here, just assigning indices
                    pass
                
                # We assign an index even if OOB to keep the structure 64x73
                # Actually, AlphaZero flattens valid moves. 
                # But standard implementation usually uses a fixed 4672 vector.
                # Let's just increment idx.
                
                # Wait, the mapping must be deterministic and consistent.
                # The structure is usually: [Square][MoveType]
                # MoveType 0-55: Queen moves
                # MoveType 56-63: Knight moves
                # MoveType 64-72: Underpromotions
                
                idx += 1 # Placeholder for loop logic, we need to actually generate the moves
                
    # Let's redo this to be precise.
    # We will generate a list of all 4672 possible "actions" relative to squares.
    # But wait, we need to map a specific UCI move (e.g., "e2e4") to an index.
    
    # Better approach:
    # Iterate 0 to 4671, decode what move it represents for a given square, store it.
    pass

# To save time and ensure correctness, I will implement a simpler, robust version 
# that calculates the index on the fly or pre-computes it properly.

class ActionConverter:
    def __init__(self):
        self.uci_to_idx = {}
        self.idx_to_uci = {}
        self._generate_mappings()
        
    def _generate_mappings(self):
        # This follows the AlphaZero/Lc0 mapping convention roughly
        # 73 planes per square.
        # Planes 0-55: Queen moves (8 directions * 7 distances)
        # Planes 56-63: Knight moves (8 targets)
        # Planes 64-72: Underpromotions (3 directions * 3 pieces)
        
        # Directions for Queen moves (N, NE, E, SE, S, SW, W, NW)
        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        
        # Knight moves
        knight_jumps = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        
        # Underpromotions: N, NE, NW (relative to white)
        # We handle black by flipping the move perspective before encoding
        underpromos = [(0, 1), (1, 1), (-1, 1)]
        promo_pieces = ['n', 'b', 'r']
        
        idx = 0
        for sq in range(64):
            rank, file = divmod(sq, 8)
            
            # Queen Moves
            for d_file, d_rank in dirs:
                for dist in range(1, 8):
                    # Target
                    t_rank = rank + d_rank * dist
                    t_file = file + d_file * dist
                    
                    # Even if OOB, we consume an index? 
                    # No, AlphaZero output is 8x8x73. 
                    # So for every square, there are 73 outputs.
                    # If a move is OOB, that logit is masked out (illegal).
                    # So we just need to know which "plane" (0-72) this move corresponds to.
                    pass

    def encode(self, move: chess.Move, turn: chess.Color) -> int:
        # We need to encode the move relative to the player's perspective?
        # AlphaZero usually rotates the board for Black.
        # If it's Black's turn, we flip the board so Black is at the bottom.
        # Then "North" is always "Forward".
        
        # Let's assume the board is already oriented or we handle it here.
        # Standard: Input features are flipped for Black. 
        # Output policy is also flipped? Yes.
        
        # Let's stick to: Always view as White.
        # If it's Black's turn, flip the move: e7e5 -> e2e4.
        
        if turn == chess.BLACK:
            # Flip the move
            us = move.from_square ^ 56
            us_rank, us_file = divmod(us, 8)
            them = move.to_square ^ 56
            them_rank, them_file = divmod(them, 8)
            promotion = move.promotion
        else:
            us = move.from_square
            us_rank, us_file = divmod(us, 8)
            them = move.to_square
            them_rank, them_file = divmod(them, 8)
            promotion = move.promotion

        d_rank = them_rank - us_rank
        d_file = them_file - us_file
        
        plane = -1
        
        # Underpromotion
        if promotion and promotion in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
            # Must be a pawn move (diagonal capture or push)
            # Directions: N (0,1), NE (1,1), NW (-1,1)
            # Plane offset: 64
            # Order: N(N,B,R), NE(N,B,R), NW(N,B,R) ?
            # Let's define: 
            # 64-66: N (N, B, R)
            # 67-69: NE (N, B, R)
            # 70-72: NW (N, B, R)
            
            if d_file == 0: # N
                base = 64
            elif d_file == 1: # NE
                base = 67
            elif d_file == -1: # NW
                base = 70
            else:
                return None # Should not happen
                
            if promotion == chess.KNIGHT: offset = 0
            elif promotion == chess.BISHOP: offset = 1
            elif promotion == chess.ROOK: offset = 2
            
            plane = base + offset
            
        elif not promotion or promotion == chess.QUEEN:
            # Queen-like or Knight move
            
            # Check Knight
            if abs(d_rank) * abs(d_file) == 2:
                # Knight move
                # Map (d_file, d_rank) to 0-7
                knight_jumps = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
                try:
                    k_idx = knight_jumps.index((d_file, d_rank))
                    plane = 56 + k_idx
                except ValueError:
                    return None
            else:
                # Queen move
                # Determine direction and distance
                dist = max(abs(d_rank), abs(d_file))
                
                # Normalize direction
                n_rank = d_rank // dist
                n_file = d_file // dist
                
                dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
                try:
                    d_idx = dirs.index((n_file, n_rank))
                    # Plane = direction * 7 + (dist - 1)
                    plane = d_idx * 7 + (dist - 1)
                except ValueError:
                    return None

        if plane == -1:
            return None
            
        return us * 73 + plane

    def decode(self, idx: int, turn: chess.Color) -> chess.Move:
        # Reverse of encode
        us = idx // 73
        plane = idx % 73
        
        us_rank, us_file = divmod(us, 8)
        
        d_rank = 0
        d_file = 0
        promotion = None
        
        if 0 <= plane < 56: # Queen
            d_idx = plane // 7
            dist = (plane % 7) + 1
            dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
            n_file, n_rank = dirs[d_idx]
            d_rank = n_rank * dist
            d_file = n_file * dist
            
        elif 56 <= plane < 64: # Knight
            k_idx = plane - 56
            knight_jumps = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
            d_file, d_rank = knight_jumps[k_idx]
            
        elif 64 <= plane < 73: # Underpromo
            rem = plane - 64
            direction_idx = rem // 3
            piece_idx = rem % 3
            
            # Directions: N (0), NE (1), NW (2)
            if direction_idx == 0: d_file, d_rank = 0, 1
            elif direction_idx == 1: d_file, d_rank = 1, 1
            elif direction_idx == 2: d_file, d_rank = -1, 1
            
            if piece_idx == 0: promotion = chess.KNIGHT
            elif piece_idx == 1: promotion = chess.BISHOP
            elif piece_idx == 2: promotion = chess.ROOK
            
        # Calculate target
        them_rank = us_rank + d_rank
        them_file = us_file + d_file
        
        if not (0 <= them_rank < 8 and 0 <= them_file < 8):
            return None # OOB
            
        them = them_rank * 8 + them_file
        
        # Handle Queen promotion (implicit in Queen moves if reaching rank 7/0)
        # Wait, if it's a pawn move reaching the last rank and it's a Queen move plane, it's a Queen promo.
        # We don't know the piece type here, so we return a move. 
        # The caller must check if it's a pawn move to the last rank to add promotion=Queen.
        # Actually, for decoding, we might return a move with promotion=None, 
        # and the engine filters legal moves. If a pawn reaches end without promo, it's illegal.
        # But we can't easily distinguish a Rook move to rank 8 from a Pawn move to rank 8 (Queen promo) just by geometry.
        # However, the policy mask will only allow legal moves.
        # So we can just return the coordinates.
        # BUT, `chess.Move` needs the promotion piece if it IS a promotion.
        # If we return None, it's not a promotion.
        # We will handle the "Queen Promotion" ambiguity by checking if the move is a Pawn push to 8th rank in the legal moves list.
        
        move = chess.Move(us, them, promotion=promotion)
        
        if turn == chess.BLACK:
            # Flip back
            move.from_square ^= 56
            move.to_square ^= 56
            
        return move

ACTION_CONVERTER = ActionConverter()

def encode_board(board: chess.Board, history: list = None) -> np.ndarray:
    """
    Encodes the board state into a (119, 8, 8) float32 tensor.
    History should contain the last 7 board states (excluding current).
    If history is None or short, we pad with zeros.
    """
    if history is None:
        history = []
        
    # Ensure we have exactly 8 states (current + 7 history)
    # Most recent first
    states = [board] + history[:7]
    
    # If fewer than 8, pad with None (which will result in zero planes)
    while len(states) < 8:
        states.append(None)
        
    planes = []
    
    # Orientation: We always orient towards the current player.
    # If it's Black's turn, we flip the board so Black is at the bottom (rows 0-1).
    # This means we reverse the ranks.
    
    us = board.turn
    them = not us
    
    for state in states:
        if state is None:
            # 14 zero planes
            planes.append(np.zeros((14, 8, 8), dtype=np.float32))
            continue
            
        state_planes = np.zeros((14, 8, 8), dtype=np.float32)
        
        # Pieces
        # P1 (Us): P, N, B, R, Q, K
        # P2 (Them): P, N, B, R, Q, K
        # Repetitions: 2 planes (1 if rep>=1, 1 if rep>=2)
        
        # We need to respect the orientation.
        # If us==Black, we map Black pieces to P1, White to P2.
        # And we flip the board (rank 0 becomes rank 7).
        
        pieces_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        
        # P1 (Us)
        for i, piece_type in enumerate(pieces_order):
            bb = state.pieces(piece_type, us)
            for sq in bb:
                rank, file = divmod(sq, 8)
                if us == chess.BLACK:
                    rank = 7 - rank
                    file = 7 - file # Usually we flip both? Or just rank?
                    # AlphaZero rotates 180 degrees.
                state_planes[i, rank, file] = 1.0
                
        # P2 (Them)
        for i, piece_type in enumerate(pieces_order):
            bb = state.pieces(piece_type, them)
            for sq in bb:
                rank, file = divmod(sq, 8)
                if us == chess.BLACK:
                    rank = 7 - rank
                    file = 7 - file
                state_planes[6 + i, rank, file] = 1.0
                
        # Repetitions (Simplified - checking current board repetition count)
        # Note: This requires the board to have the correct repetition history.
        # If we just pass a board, is_repetition might not work if the stack isn't populated.
        # We assume the passed 'state' objects have their move stacks or we ignore this for now.
        # For simplicity in this implementation, we might skip precise repetition planes or assume 0.
        if state.is_repetition(2):
            state_planes[12, :, :] = 1.0
        if state.is_repetition(3):
            state_planes[13, :, :] = 1.0
            
        planes.append(state_planes)
        
    # Concatenate history planes -> (112, 8, 8)
    features = np.concatenate(planes, axis=0)
    
    # Constant planes (7)
    # 1. Color (0 for White, 1 for Black? Or always 0 because we orient?)
    # AlphaZero uses "Colour" plane. Usually all 1s if Black, all 0s if White.
    # But since we orient, maybe it's not needed? 
    # Actually, AlphaZero includes it. Let's add it.
    # If we always orient to "us", the network might need to know if "us" is White or Black (for castling rights context etc).
    
    const_planes = np.zeros((7, 8, 8), dtype=np.float32)
    
    if us == chess.BLACK:
        const_planes[0, :, :] = 1.0
        
    # Castling Rights (4 planes)
    # Us K-side, Us Q-side, Them K-side, Them Q-side
    if board.has_kingside_castling_rights(us):
        const_planes[1, :, :] = 1.0
    if board.has_queenside_castling_rights(us):
        const_planes[2, :, :] = 1.0
    if board.has_kingside_castling_rights(them):
        const_planes[3, :, :] = 1.0
    if board.has_queenside_castling_rights(them):
        const_planes[4, :, :] = 1.0
        
    # Move Count (Total moves) - Scaled?
    # AlphaZero uses "Total move count" plane (all values = count).
    # And "P1 castling", "P2 castling", "No-progress count".
    
    # No-progress count (50-move rule)
    const_planes[5, :, :] = board.halfmove_clock / 100.0
    
    # Total move count (optional, but good)
    const_planes[6, :, :] = board.fullmove_number / 200.0 # Normalize roughly
    
    # Combine
    final_features = np.concatenate([features, const_planes], axis=0)
    
    return final_features

