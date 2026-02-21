#ifndef MAT_H
#define MAT_H

#include <cstddef>
#include <utility>

class Mat {
public:
    std::size_t rows;
    std::size_t cols;

    Mat(std::size_t t_rows, std::size_t t_cols);
    Mat(std::size_t t_rows, std::size_t t_cols, float t_value);
    
    ~Mat();
    
    Mat(const Mat& t_other);
    
    Mat(Mat&& t_other) noexcept;
    
    Mat& operator=(const Mat& t_other);
    
    Mat& operator=(Mat&& t_other) noexcept;

    std::pair<std::size_t, std::size_t> shape() const;
    std::size_t size() const;
    
    float& at(std::size_t row, std::size_t col);
    const float& at(std::size_t row, std::size_t col) const;
    
    float& operator()(std::size_t row, std::size_t col);
    const float& operator()(std::size_t row, std::size_t col) const;
    
    float* data() { return m_data; }
    const float* data() const { return m_data; }

private:
    float* m_data;
};

#endif