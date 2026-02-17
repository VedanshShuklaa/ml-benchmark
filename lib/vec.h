#ifndef VEC_H
#define VEC_H

#include <cstdint>
#include <iostream>

class Vec {
    size_t m_size;
    float* m_data;

public:

    Vec(size_t t_size);
    Vec(size_t t_size, float t_value);

    ~Vec() noexcept;
    Vec(const Vec& t_other);
    Vec& operator=(const Vec& t_other);
    float& operator[](size_t t_index);

    const float& operator[](size_t t_index) const;
    size_t size() const;
    float* data() const;

    friend std::ostream& operator<<(std::ostream& os, const Vec& t_v);
};

size_t align_size(size_t t_size);

#endif // VEC_H
