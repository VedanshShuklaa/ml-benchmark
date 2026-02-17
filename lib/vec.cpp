#include <iostream>
#include <cstdint>
#include <sys/types.h>
#include "vec.h"

size_t align_size(size_t t_size) {
    return (t_size + 31) & ~31; // Align to 32-byte boundary
}

Vec::Vec(size_t t_size) {
    m_size = t_size;
    m_data = static_cast<float*>(
        std::aligned_alloc(32, align_size(m_size * sizeof(float)))
    );
    if (!m_data)
        throw std::bad_alloc();
};

Vec::Vec(size_t t_size, float t_value) {
    m_size = t_size;
    m_data = static_cast<float*>(
        std::aligned_alloc(32, align_size(m_size * sizeof(float)))
    );
    if (!m_data)
        throw std::bad_alloc();
    for(size_t i = 0; i < m_size; i++) {
        m_data[i] = t_value;
    }
}

Vec::~Vec() noexcept {
    std::free(m_data);
}

Vec::Vec(const Vec& t_other)
    : m_size(t_other.m_size)
{
    if (m_size == 0) {
        m_data = nullptr;
        return;
    }
    m_data = static_cast<float*>(
        std::aligned_alloc(32, align_size(m_size * sizeof(float)))
    );
    if (!m_data)
        throw std::bad_alloc();
    std::copy(t_other.m_data, t_other.m_data + m_size, m_data);
}

Vec& Vec::operator=(const Vec& t_other) {
    if(this != &t_other) {
        std::free(m_data);
        m_size = t_other.m_size;
        m_data = static_cast<float*>(
            std::aligned_alloc(32, align_size(m_size * sizeof(float)))
        );
        if (!m_data)
            throw std::bad_alloc();
        for(size_t i = 0; i < m_size; i++) {
            m_data[i] = t_other[i];
        }
    }
    return *this;
}

float& Vec::operator[](size_t t_index) {
    return m_data[t_index];
}

const float& Vec::operator[](size_t t_index) const {
    return m_data[t_index];
}

size_t Vec::size() const {
    return m_size;
}

float* Vec::data() const {
    return m_data;
}

std::ostream& operator<<(std::ostream& os, const Vec& t_v) {
    for(size_t i = 0; i < t_v.m_size; i++) {
        os << t_v.m_data[i] << " ";
    }
    return os;
}