#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <set>
#include <vector>


namespace py = pybind11;


std::vector<int> calc_greedy_assignment(py::array_t<int> cost_matrix_np) {
    // Convert numpy array to C++ vector of vectors
    py::buffer_info buf = cost_matrix_np.request();
    int num_rows = buf.shape[0];
    int num_cols = buf.shape[1];
    auto *ptr = static_cast<int *>(buf.ptr);
    std::vector<std::vector<int>> cost_matrix(num_rows, std::vector<int>(num_cols));

    for (int i = 0; i < num_rows; ++i)
        for (int j = 0; j < num_cols; ++j)
            cost_matrix[i][j] = ptr[i * num_cols + j];

    // Initialize sorted rows and columns
    std::vector<std::set<std::pair<int, int>>> sorted_rows(num_rows);
    std::vector<std::set<std::pair<int, int>>> sorted_cols(num_cols);
    std::vector<bool> rowNotDeleted(num_rows, true);
    std::vector<bool> colNotDeleted(num_cols, true);
    std::vector<int> node_assignment(num_rows);

    // Populate sorted_rows and sorted_cols
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            sorted_rows[i].insert(std::make_pair(cost_matrix[i][j], j));
            sorted_cols[j].insert(std::make_pair(cost_matrix[i][j], i));
        }
    }

    // Greedy assignment
    for (int k = 0; k < num_rows; k++) {
        int min_cost = 100, min_row = -1, min_col = -1;
        for (int i = 0; i < num_rows; i++) {
            if (rowNotDeleted[i] && !sorted_rows[i].empty()) {
                auto [cost, col] = *sorted_rows[i].begin();
                if (cost < min_cost) {
                    min_cost = cost;
                    min_row = i;
                    min_col = col;
                }
            }
        }
        for (int i = 0; i < num_cols; i++) {
            if (colNotDeleted[i] && !sorted_cols[i].empty()) {
                auto [cost, row] = *sorted_cols[i].begin();
                if (cost < min_cost) {
                    min_cost = cost;
                    min_row = row;
                    min_col = i;
                }
            }
        }
        node_assignment[min_row] = min_col;
        rowNotDeleted[min_row] = false;
        colNotDeleted[min_col] = false;
        for (int i = 0; i < num_rows; ++i) {
            if (rowNotDeleted[i]) {
                sorted_rows[i].erase({cost_matrix[i][min_col], min_col});
            }
        }
        for (int j = 0; j < num_cols; ++j) {
            if (colNotDeleted[j]) {
                sorted_cols[j].erase({cost_matrix[min_row][j], min_row});
            }
        }
    }
    return node_assignment;
}

// Pybind11 module definition
PYBIND11_MODULE(greedy_assignment, m) {
    m.def("calc_greedy_assignment", &calc_greedy_assignment, "Calculate greedy assignment");
}