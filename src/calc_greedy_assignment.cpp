#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <set>
#include <vector>


namespace py = pybind11;


std::vector<int> calc_greedy_assignment(py::array_t<double> cost_matrix_np) {
    // Convert numpy array to C++ vector of vectors
    py::buffer_info buf = cost_matrix_np.request();
    int num_rows = buf.shape[0];
    int num_cols = buf.shape[1];
    auto *ptr = static_cast<double *>(buf.ptr);

    // Flatten the cost matrix
    std::vector<std::vector<double>> cost_matrix(num_rows, std::vector<double>(num_cols));

    for (int i = 0; i < num_rows; ++i)
        for (int j = 0; j < num_cols; ++j)
            cost_matrix[i][j] = ptr[i * num_cols + j];

    // Initialize sorted rows and columns
    std::vector<std::set<std::pair<double, int>>> sorted_rows(num_rows);
    std::vector<std::set<std::pair<double, int>>> sorted_cols(num_cols);
    std::vector<bool> rowNotDeleted(num_rows, true);
    std::vector<bool> colNotDeleted(num_cols, true);
    std::vector<int> node_assignment(num_rows, -1);

    // Populate sorted_rows and sorted_cols
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            sorted_rows[i].insert(std::make_pair(cost_matrix[i][j], j));
            sorted_cols[j].insert(std::make_pair(cost_matrix[i][j], i));
        }
    }

    // Greedy assignment
    for (int k = 0; k < num_rows; k++) {
        double min_cost = std::numeric_limits<double>::max();
        int min_row = -1, min_col = -1;
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


std::vector<int> calc_greedy_assignment_fast(py::array_t<double> cost_matrix_np) {
    // Access the numpy array as a 1D buffer for faster indexing
    py::buffer_info buf = cost_matrix_np.request();
    int num_rows = buf.shape[0];
    int num_cols = buf.shape[1];
    auto *ptr = static_cast<double *>(buf.ptr);
    
    // Flattened cost matrix
    // std::vector<double> cost_matrix(ptr, ptr + (num_rows * num_cols));

    // Structures to keep track of row and column availability
    std::vector<bool> rowNotDeleted(num_rows, true);
    std::vector<bool> colNotDeleted(num_cols, true);
    std::vector<int> node_assignment(num_rows, -1);
    double tmp_cost, min_cost;
    int min_row, min_col;

    // Greedy assignment loop
    for (int k = 0; k < num_rows; ++k) {
        min_cost = 100.0;
        min_row = -1, min_col = -1;

        // Find the minimum cost element in the available rows and columns
        for (int i = 0; i < num_rows; ++i) {
            if (rowNotDeleted[i]) {
                for (int j = 0; j < num_cols; ++j) {
                    if (colNotDeleted[j]) {
                        // tmp_cost = cost_matrix[i * num_cols + j];
                        tmp_cost = ptr[i * num_cols + j];
                        if (tmp_cost < min_cost) {
                            min_cost = tmp_cost;
                            min_row = i;
                            min_col = j;
                        }
                    }
                }
            }
        }

        // Assign minimum cost element's column to the row
        node_assignment[min_row] = min_col;
        rowNotDeleted[min_row] = false;
        colNotDeleted[min_col] = false;
    }

    return node_assignment;
}


// Pybind11 module definition
PYBIND11_MODULE(greedy_assignment, m) {
    m.def("calc_greedy_assignment", &calc_greedy_assignment, "Calculate greedy assignment.");
    m.def("calc_greedy_assignment_fast", &calc_greedy_assignment_fast, "Calculate greedy assignment with a more efficient implementation.");
}