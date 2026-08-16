//
// Created by twqb on 8/16/26.
//
#include <armadillo>

#ifndef NAIVE_INFER_LOAD_DATA_HPP
#define NAIVE_INFER_LOAD_DATA_HPP

namespace naive_infer {
  class CSVDataLoader {
  public:
    /**
     * Initializes a matrix from a CSV file
     * @param file_path Path to the CSV file
     * @param split_char Delimiter character
     * @return fmat loaded from the CSV file
     */
    static arma::fmat LoadData(const std::string &file_path, char split_char = ',');

  private:
    /**
     * Obtains the matrix dimensions of the CSV file.
     * Used by LoadData to initialize the returned fmat.
     * @param file Input stream of the CSV file
     * @param split_char Delimiter character
     * @return Pair (rows, cols) representing matrix size
     */
    static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
  };
}

#endif //NAIVE_INFER_LOAD_DATA_HPP