#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "chess-library/include/chess.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <map>
#include <random>

namespace py = pybind11;
using namespace chess;

// Node structure
struct Node {
    int visit_count = 0;
    double value_sum = 0.0;
    double prior = 0.0;
    std::map<uint16_t, Node*> children; // Move encoded as uint16_t
    bool is_expanded = false;
    
    ~Node() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }
    
    double value() const {
        if (visit_count == 0) return 0.0;
        return value_sum / visit_count;
    }
};

class MCTS {
public:
    MCTS(py::object model, int num_simulations, double c_puct) 
        : model(model), num_simulations(num_simulations), c_puct(c_puct) {}

    py::tuple search(std::string fen) {
        Board board(fen);
        Node* root = new Node();
        
        // Expand root
        _expand(root, board);
        _add_noise(root);
        
        for (int i = 0; i < num_simulations; ++i) {
            Node* node = root;
            Board current_board = board; // Copy
            std::vector<Node*> search_path;
            search_path.push_back(node);
            
            // Selection
            while (node->is_expanded && !node->children.empty()) {
                uint16_t move_int = _select_child(node);
                Move move = Move(move_int);
                
                current_board.makeMove(move);
                node = node->children[move_int];
                search_path.push_back(node);
            }
            
            // Expansion & Evaluation
            double value = 0.0;
            // Check terminal
            auto status = current_board.isGameOver();
            if (status.second != GameResult::NONE) {
                // Terminal
                if (status.second == GameResult::DRAW) value = 0.0;
                else {
                     // If current player lost (previous player moved to win), value is -1?
                     // Wait, isGameOver returns result.
                     // If White won and it's Black's turn, value for Black is -1.
                     // If White won and it's White's turn (impossible), value is 1.
                     // Usually we check result relative to side to move.
                     // If side to move is lost, value is -1.
                     value = -1.0; 
                }
            } else {
                if (!node->is_expanded) {
                    value = _expand_and_evaluate(node, current_board);
                } else {
                    // Already expanded but no children (stalemate/checkmate handled above?)
                    // If no children but not game over? (Should not happen if move gen is correct)
                    value = 0.0;
                }
            }
            
            // Backprop
            for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
                Node* n = *it;
                n->visit_count++;
                n->value_sum += value;
                value = -value;
            }
        }
        
        // Select best move (most visits)
        uint16_t best_move_int = 0;
        int max_visits = -1;
        
        // Also build policy vector for return
        // We need to map moves to indices 0-4671.
        // This is hard to do exactly same as Python without the lookup table.
        // For now, we return the best move string and the visit counts as a map?
        // Or we just return best move for playing.
        // For training, we need the full policy.
        
        // Let's return (best_move_uci, policy_probs_list)
        // We need to implement the move->index mapping in C++ to match Python.
        // Or we return a list of (move_uci, visit_count) and let Python build the vector.
        
        std::vector<std::pair<std::string, int>> visits;
        
        for (auto& pair : root->children) {
            if (pair.second->visit_count > max_visits) {
                max_visits = pair.second->visit_count;
                best_move_int = pair.first;
            }
            visits.push_back({uci::moveToUci(Move(pair.first)), pair.second->visit_count});
        }
        
        // Cleanup
        // In real engine, we might keep the tree.
        delete root;
        
        return py::make_tuple(uci::moveToUci(Move(best_move_int)), visits);
    }

private:
    py::object model;
    int num_simulations;
    double c_puct;
    
    void _expand(Node* node, Board& board) {
        Movelist moves;
        movegen::legalmoves(moves, board);
        
        if (moves.empty()) {
            node->is_expanded = true;
            return;
        }
        
        for (const auto& move : moves) {
            node->children[move.move()] = new Node();
        }
        node->is_expanded = true;
    }
    
    void _add_noise(Node* node) {
        // TODO: Implement Dirichlet noise
    }
    
    uint16_t _select_child(Node* node) {
        double best_score = -1e9;
        uint16_t best_move = 0;
        
        for (auto& pair : node->children) {
            Node* child = pair.second;
            double q = child->value();
            // Inverse value for selection? No, child value is from opponent perspective.
            // So we want to minimize opponent value?
            // Standard AlphaZero: Q(s,a) is expected return for current player.
            // Child stores value for NEXT player.
            // So Q = -child->value().
            double q_val = -child->value();
            
            double u = c_puct * child->prior * std::sqrt(node->visit_count) / (1 + child->visit_count);
            double score = q_val + u;
            
            if (score > best_score) {
                best_score = score;
                best_move = pair.first;
            }
        }
        return best_move;
    }
    
    double _expand_and_evaluate(Node* node, Board& board) {
        // 1. Encode board
        // We call Python for now to be safe and compatible
        // py::object py_board = chess.Board(board.getFen())
        // tensor = encode_board(py_board)
        
        // This is the bottleneck we wanted to avoid.
        // But implementing 119 planes in C++ correctly without bugs in 1 shot is hard.
        // Let's try to call Python encode_board.
        // Even with this call, the TREE SEARCH (selection, move gen) is C++.
        // That is 90% of the work in MCTS (simulations).
        // The evaluation happens once per leaf.
        // The selection happens depth times.
        // So C++ move gen is a big win.
        
        py::gil_scoped_acquire acquire;
        py::module_ chess_mod = py::module_::import("chess");
        py::object py_board = chess_mod.attr("Board")(board.getFen());
        
        py::module_ utils = py::module_::import("src.chess_utils");
        py::object tensor = utils.attr("encode_board")(py_board);
        
        // 2. Predict
        // tensor is numpy array.
        // If model is ModelServer wrapper, it handles it.
        // If model is PyTorch module, we need to wrap it or ensure it accepts numpy.
        // Our updated MCTS._evaluate handles this.
        // We can just call model(tensor).
        
        py::object result = model(tensor); // Returns (p, v)
        // If result is future?
        if (py::hasattr(result, "result")) {
            result = result.attr("result")();
        }
        
        py::tuple res_tuple = result.cast<py::tuple>();
        py::object policy_logits = res_tuple[0];
        double value = res_tuple[1].cast<double>();
        
        // 3. Assign priors
        // We need to map policy logits to moves.
        // This is tricky. Python has ACTION_CONVERTER.
        // We should call ACTION_CONVERTER.decode(idx) -> move
        // Or better: Iterate legal moves, encode them -> idx, get logit.
        
        py::object converter = utils.attr("ACTION_CONVERTER");
        
        // Get softmax
        // We can do softmax in C++ or Python.
        // Let's assume logits are numpy.
        // We need to map moves to indices.
        
        // Iterate children (legal moves)
        for (auto& pair : node->children) {
            Move move(pair.first);
            std::string uci = uci::moveToUci(move);
            py::object py_move = chess_mod.attr("Move").attr("from_uci")(uci);
            
            // encode(move, turn)
            py::object idx_obj = converter.attr("encode")(py_move, py_board.attr("turn"));
            
            if (!idx_obj.is_none()) {
                int idx = idx_obj.cast<int>();
                // Get logit
                // Assuming policy_logits is 1D array
                // We access it via buffer protocol or cast
                // py::array_t<float> logits = policy_logits.cast<py::array_t<float>>();
                // But casting every time is slow.
                // Just use attr item()
                double logit = policy_logits.attr("item")(idx).cast<double>();
                pair.second->prior = std::exp(logit); // Softmax numerator
            }
        }
        
        // Normalize priors
        double sum_priors = 0.0;
        for (auto& pair : node->children) sum_priors += pair.second->prior;
        if (sum_priors > 0) {
            for (auto& pair : node->children) pair.second->prior /= sum_priors;
        }
        
        return value;
    }
};
