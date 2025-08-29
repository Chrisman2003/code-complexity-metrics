#ifndef POLYHEDRAL_GRAVITY_PARALLEL_COMMON_H
#define POLYHEDRAL_GRAVITY_PARALLEL_COMMON_H

#if defined(__clang__) && defined(__CUDA__)
#define FUNC_PREFIX __device__ __host__
#define CTOR_PREFIX __device__ __host__
#elif defined(_OPENACC)
#define FUNC_PREFIX _Pragma("acc routine")
#define CTOR_PREFIX
#else
#define FUNC_PREFIX
#define CTOR_PREFIX
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <common.h>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

constexpr double EPSILON_ZERO_OFFSET = 1e-14;
constexpr double EPSILON_ALMOST_EQUAL = 1e-10;

constexpr double PI =
        3.1415926535897932384626433832795028841971693993751058209749445923;
constexpr double PI2 =
        6.2831853071795864769252867665590057683943387987502116419498891846;
constexpr double PI_2 =
        1.5707963267948966192313216916397514420985846996875529104874722961;

/**
 * The gravitational constant G in [m^3/(kg*s^2)].
 * @related in his paper above Equation (4)C
 */
constexpr double GRAVITATIONAL_CONSTANT = 6.67430e-11;

#if FLOAT_BITS == 32
using FloatType = float;
#elif FLOAT_BITS == 64
using FloatType = double;
#elif FLOAT_BITS == 128
using FloatType = __float128;
#else
#error "Invliad float bits size"
#endif


FUNC_PREFIX
inline bool almostEqualRelative(FloatType lhs, FloatType rhs,
                                double epsilon = EPSILON_ALMOST_EQUAL) {
    const FloatType diff = std::abs(rhs - lhs);
    const FloatType largerValue = std::max(std::abs(rhs), std::abs(lhs));
    return diff <= largerValue * epsilon;
}

template<typename T = FloatType>
struct Array3Base {
    T data[3];

    constexpr Array3Base()
        : data{{}, {}, {}} {
    }
    constexpr Array3Base(const T x, const T y, const T z)
        : data{x, y, z} {
    }

    FUNC_PREFIX Array3Base<T> operator+(const Array3Base<T> &rhs) const {
        return {data[0] + rhs.data[0], data[1] + rhs.data[1], data[2] + rhs.data[2]};
    }
    FUNC_PREFIX Array3Base<T> operator-(const Array3Base<T> &rhs) const {
        return {data[0] - rhs.data[0], data[1] - rhs.data[1], data[2] - rhs.data[2]};
    }
    FUNC_PREFIX Array3Base<T> operator*(const Array3Base<T> &rhs) const {
        return {data[0] * rhs.data[0], data[1] * rhs.data[1], data[2] * rhs.data[2]};
    }
    FUNC_PREFIX Array3Base<T> operator/(T scalar) const {
        return {data[0] / scalar, data[1] / scalar, data[2] / scalar};
    }
    FUNC_PREFIX Array3Base<T> operator*(T scalar) const {
        return {data[0] * scalar, data[1] * scalar, data[2] * scalar};
    }
    FUNC_PREFIX T &operator[](const size_t index) {
        return data[index];
    }
    FUNC_PREFIX T operator[](const size_t index) const {
        return data[index];
    }
};

template<typename T = FloatType>
struct Array6Base {
    T data[6];
    constexpr Array6Base()
        : data{} {
    }
    constexpr Array6Base(const T v1, const T v2, const T v3, const T v4, const T v5, const T v6)
        : data{v1, v2, v3, v4, v5, v6} {
    }
    FUNC_PREFIX Array6Base operator+(const Array6Base &rhs) const {
        return {data[0] + rhs.data[0], data[1] + rhs.data[1], data[2] + rhs.data[2], data[3] + rhs.data[3], data[4] + rhs.data[4], data[5] + rhs.data[5]};
    }
    FUNC_PREFIX Array6Base operator*(const T scalar) const {
        return {data[0] * scalar, data[1] * scalar, data[2] * scalar,
                data[3] * scalar, data[4] * scalar, data[5] * scalar};
    }
    FUNC_PREFIX T operator[](const size_t index) const {
        return data[index];
    }
};

using Array3 = Array3Base<FloatType>;
using Array6 = Array6Base<FloatType>;
using IndexArray3 = Array3Base<size_t>;
using Array3Triplet = Array3Base<Array3Base<FloatType>>;

struct GravityModelResult {
    FloatType potential;
    Array3 acceleration;
    Array6 gradiometricTensor;

    CTOR_PREFIX GravityModelResult()
        : potential(0), acceleration{}, gradiometricTensor{} {
    }
    CTOR_PREFIX GravityModelResult(const FloatType _potential, const Array3 &_acceleration, const Array6 &_gradiometricTensor)
        : potential(_potential), acceleration(_acceleration),
          gradiometricTensor(_gradiometricTensor) {
    }


    FUNC_PREFIX GravityModelResult &operator+=(const GravityModelResult &rhs) {
        potential += rhs.potential;
        acceleration = acceleration + rhs.acceleration;
        gradiometricTensor = gradiometricTensor + rhs.gradiometricTensor;

        return *this;
    }

    FUNC_PREFIX GravityModelResult operator+(const GravityModelResult &rhs) const {
        return {potential + rhs.potential, acceleration + rhs.acceleration, gradiometricTensor + rhs.gradiometricTensor};
    }
};

struct Singularity {
    FloatType a;
    Array3 b;

    CTOR_PREFIX Singularity()
        : a{}, b{} {
    }
    CTOR_PREFIX Singularity(const FloatType _a, const Array3 &_b)
        : a{_a}, b{_b} {
    }
};

struct Distance {
    FloatType l1;
    FloatType l2;
    FloatType s1;
    FloatType s2;
};

struct TranscendentalExpression {
    FloatType ln;
    FloatType an;
};

struct HessianPlane {
    FloatType a;
    FloatType b;
    FloatType c;
    FloatType d;
};

using Matrix = Array3Base<Array3Base<FloatType>>;

#if FLOAT_BITS == 128
FUNC_PREFIX inline FloatType euclideanNorm(const Array3 &a) {
    return __builtin_sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
#else
FUNC_PREFIX inline FloatType euclideanNorm(const Array3 &a) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
#endif


FUNC_PREFIX inline FloatType det(const Matrix &matrix) {
    return matrix[0][0] * matrix[1][1] * matrix[2][2] +
           matrix[0][1] * matrix[1][2] * matrix[2][0] +
           matrix[0][2] * matrix[1][0] * matrix[2][1] -
           matrix[0][2] * matrix[1][1] * matrix[2][0] -
           matrix[0][0] * matrix[1][2] * matrix[2][1] -
           matrix[0][1] * matrix[1][0] * matrix[2][2];
}

FUNC_PREFIX inline Matrix transpose(const Matrix &matrix) {
    Matrix transposed;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            transposed[i][j] = matrix[j][i];
        }
    }
    return transposed;
}

FUNC_PREFIX inline Array3 cross(const Array3 &lhs,
                                const Array3 &rhs) {
    Array3 result{};
    result[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
    result[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
    result[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];
    return result;
}

FUNC_PREFIX inline Array3 normal(const Array3 &first,
                                 const Array3 &second) {
    const Array3 crossProduct = cross(first, second);
    const auto norm = euclideanNorm(crossProduct);
    return crossProduct / norm;
}

FUNC_PREFIX inline FloatType dot(const Array3 &lhs, const Array3 &rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

FUNC_PREFIX
inline int sgn(FloatType val, FloatType cutoffEpsilon = EPSILON_ZERO_OFFSET) {
    return val < -cutoffEpsilon ? -1 : val > cutoffEpsilon ? 1
                                                           : 0;
}

/**
 * Calculates the magnitude between two values and return true if the magnitude
 * between the exponents in greater than 17.
 * @tparam T numerical type
 * @param first first number
 * @param second second number
 * @return true if the difference is too be huge, so that floating point
 * absorption will happen
 */
FUNC_PREFIX inline bool isCriticalDifference(const FloatType &first, const FloatType &second) {
    // 50 is the (log2) exponent of the floating point (1 / 1e-15)
    constexpr int maxExponentDifference = 50;
    int x, y;

#if FLOAT_BITS == 128
    __builtin_frexpf(first, &x);
    __builtin_frexpf(second, &y);
#else
    std::frexp(first, &x);
    std::frexp(second, &y);
#endif
    return std::abs(x - y) > maxExponentDifference;
}

FUNC_PREFIX inline Array6 concat(const Array3 &first, const Array3 &second) {
    return {first[0], first[1], first[2], second[0], second[1], second[2]};
}

struct GlobalResources {
    GlobalResources(int &argc, char *argv[]);
    ~GlobalResources();
};

void read_tetgen(const std::string &filename_base, std::vector<Array3> &Vertices, std::vector<IndexArray3> &Faces);
void read_obj(const std::string &filename, std::vector<Array3> &Vertices, std::vector<IndexArray3> &Faces);

class GravityEvaluableBase {
public:
    GravityEvaluableBase(const std::vector<Array3> &Vertices, const std::vector<IndexArray3> &Faces, const double density)
        : _density{density}, _vertices{Vertices}, _faces{Faces}, _initialized{false} {
    }
    virtual ~GravityEvaluableBase() = default;
    virtual GravityModelResult evaluate(const Array3 &Point) {
        return {};
    };

protected:
    double _density;

    const std::vector<Array3> &_vertices;
    const std::vector<IndexArray3> &_faces;

    bool _initialized;
};

std::unique_ptr<GravityEvaluableBase> create_gravity_evaluable(
        const std::vector<Array3> &Vertices,
        const std::vector<IndexArray3> &Faces,
        double density);

#endif