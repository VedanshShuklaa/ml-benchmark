#include "mat.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

Mat::Mat(std::size_t t_rows, std::size_t t_cols) : rows(t_rows), cols(t_cols), m_data(nullptr) {
    if (rows == 0 || cols == 0) {
        rows = 0;
        cols = 0;
        return;
    }
    
    std::size_t bytes = rows * cols * sizeof(float);
    std::size_t aligned_bytes = (bytes + 15) & ~15;
    m_data = static_cast<float*>(std::aligned_alloc(16, aligned_bytes));
    
    if (!m_data) {
        throw std::bad_alloc();
    }
    
    std::memset(m_data, 0, bytes);
}

Mat::Mat(std::size_t t_rows, std::size_t t_cols, float t_value) : rows(t_rows), cols(t_cols), m_data(nullptr) {
    if (rows == 0 || cols == 0) {
        rows = 0;
        cols = 0;
        return;
    }
    
    std::size_t bytes = rows * cols * sizeof(float);
    std::size_t aligned_bytes = (bytes + 15) & ~15;
    m_data = static_cast<float*>(std::aligned_alloc(16, aligned_bytes));
    
    if (!m_data) {
        throw std::bad_alloc();
    }
    
    for (std::size_t i = 0; i < rows * cols; i++) {
        m_data[i] = t_value;
    }
}

Mat::~Mat() {
    std::free(m_data);
}

Mat::Mat(const Mat& t_other) : rows(t_other.rows), cols(t_other.cols), m_data(nullptr) {
    if (rows == 0 || cols == 0) {
        return;
    }
    
    std::size_t bytes = rows * cols * sizeof(float);
    std::size_t aligned_bytes = (bytes + 15) & ~15;
    m_data = static_cast<float*>(std::aligned_alloc(16, aligned_bytes));
    
    if (!m_data) {
        throw std::bad_alloc();
    }
    
    std::memcpy(m_data, t_other.m_data, bytes);
}

Mat::Mat(Mat&& t_other) noexcept : rows(t_other.rows), cols(t_other.cols), m_data(t_other.m_data) {
    t_other.rows = 0;
    t_other.cols = 0;
    t_other.m_data = nullptr;
}

Mat& Mat::operator=(const Mat& t_other) {
    if (this != &t_other) {
        std::free(m_data);
        m_data = nullptr;
        
        rows = t_other.rows;
        cols = t_other.cols;
        
        if (rows > 0 && cols > 0) {
            std::size_t bytes = rows * cols * sizeof(float);
            std::size_t aligned_bytes = (bytes + 15) & ~15;
            m_data = static_cast<float*>(std::aligned_alloc(16, aligned_bytes));
            
            if (!m_data) {
                rows = 0;
                cols = 0;
                throw std::bad_alloc();
            }
            
            std::memcpy(m_data, t_other.m_data, bytes);
        }
    }
    return *this;
}

Mat& Mat::operator=(Mat&& t_other) noexcept {
    if (this != &t_other) {
        std::free(m_data);
        
        rows = t_other.rows;
        cols = t_other.cols;
        m_data = t_other.m_data;
        
        t_other.rows = 0;
        t_other.cols = 0;
        t_other.m_data = nullptr;
    }
    return *this;
}

std::pair<std::size_t, std::size_t> Mat::shape() const {
    return std::make_pair(rows, cols);
}

std::size_t Mat::size() const {
    return rows * cols;
}

float& Mat::at(std::size_t row, std::size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Mat::at() index out of range");
    }
    return m_data[row * cols + col];
}

const float& Mat::at(std::size_t row, std::size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Mat::at() index out of range");
    }
    return m_data[row * cols + col];
}

float& Mat::operator()(std::size_t row, std::size_t col) {
    return m_data[row * cols + col];
}

const float& Mat::operator()(std::size_t row, std::size_t col) const {
    return m_data[row * cols + col];
}