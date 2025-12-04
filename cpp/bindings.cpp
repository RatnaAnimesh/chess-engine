#include <pybind11/pybind11.h>
#include "mcts.cpp"

namespace py = pybind11;

PYBIND11_MODULE(chess_engine_cpp, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<py::object, int, double>())
        .def("search", &MCTS::search);
}
