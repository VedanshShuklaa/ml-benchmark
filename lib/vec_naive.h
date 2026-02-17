#ifndef VEC_NAIVE_H
#define VEC_NAIVE_H

#include <cstdint>
#include <iostream>
namespace naive {
    
    class Vec {
        uint32_t m_size;
        float* m_data;

    public:

        Vec(uint32_t size);
        Vec(uint32_t size, float value);

        ~Vec();
        Vec(const Vec& other);
        Vec& operator=(const Vec& other);

        float& operator[](uint32_t index);
        const float& operator[](uint32_t index) const;

        uint32_t size() const;

        friend std::ostream& operator<<(std::ostream& os, const Vec& v);
    };

} // namespace naive

#endif // VEC_NAIVE_H
