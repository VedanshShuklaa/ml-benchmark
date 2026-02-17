#include <iostream>
#include <cstdint>
#include <sys/types.h>
#include "vec_naive.h"

namespace naive {

    Vec::Vec(uint32_t t_size) {
        m_size = t_size;
        m_data = new float[t_size];
    };

    Vec::Vec(uint32_t t_size, float t_value) {
        m_size = t_size;
        m_data = new float[t_size];
        for(uint32_t i = 0; i < m_size; i++) {
            m_data[i] = t_value;
        }
    }

    Vec::~Vec() {
        delete[] m_data;
    }

    Vec::Vec(const Vec& t_other) {
        m_size = t_other.m_size;
        m_data = new float[m_size];
        for(uint32_t i = 0; i < m_size; i++) {
            m_data[i] = t_other[i];
        }
    }

    Vec& Vec::operator=(const Vec& t_other) {
        if(this != &t_other) {
            delete[] m_data;
            m_size = t_other.m_size;
            m_data = new float[m_size];
            for(uint32_t i = 0; i < m_size; i++) {
                m_data[i] = t_other[i];
            }
        }
        return *this;
    }

    float& Vec::operator[](uint32_t t_index) {
        return m_data[t_index];
    }

    const float& Vec::operator[](uint32_t t_index) const {
        return m_data[t_index];
    }

    uint32_t Vec::size() const {
        return m_size;
    }

    std::ostream& operator<<(std::ostream& os, const Vec& v) {
        for(uint32_t i = 0; i < v.m_size; i++) {
            os << v.m_data[i] << " ";
        }
        return os;
    }
    
} // namespace naive
