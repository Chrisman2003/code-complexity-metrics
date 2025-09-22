#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <stdexcept>

// -----------------------------
// Utility templates
// -----------------------------
template<typename T>
T gcd(T a, T b) {
    while (b != 0) {
        T t = b;
        b = a % b;
        a = t;
    }
    return a;
}

template<typename T>
T lcm(T a, T b) {
    return (a / gcd(a, b)) * b;
}

template<typename T>
T factorial(T n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// -----------------------------
// Complex Data Structures
// -----------------------------
struct Node {
    int value;
    std::vector<Node*> children;
    Node(int v) : value(v) {}
};


class Graph {
    std::map<int, Node*> nodes;
public:
    ~Graph() {
        for (auto& [_, node] : nodes) {
            delete node;
        }
    }

    void addNode(int value) {
        if (nodes.find(value) == nodes.end()) {
            nodes[value] = new Node(value);
        }
    }

    void addEdge(int from, int to) {
        addNode(from);
        addNode(to);
        nodes[from]->children.push_back(nodes[to]);
    }

    int sumValuesDFS(int start) {
        std::set<int> visited;
        return dfsHelper(nodes[start], visited);
    }

private:
    int dfsHelper(Node* node, std::set<int>& visited) {
        if (!node || visited.count(node->value)) return 0;
        visited.insert(node->value);
        int sum = node->value;
        for (auto child : node->children) {
            sum += dfsHelper(child, visited);
        }
        return sum;
    }
};

// -----------------------------
// Template Utilities
// -----------------------------
template<typename T>
class Matrix {
    std::vector<std::vector<T>> data;
public:
    Matrix(size_t rows, size_t cols) : data(rows, std::vector<T>(cols, 0)) {}

    void set(size_t r, size_t c, T val) {
        if (r >= data.size() || c >= data[0].size()) throw std::out_of_range("Matrix index");
        data[r][c] = val;
    }

    T get(size_t r, size_t c) const {
        if (r >= data.size() || c >= data[0].size()) throw std::out_of_range("Matrix index");
        return data[r][c];
    }

    Matrix<T> operator+(const Matrix<T>& other) const {
        if (data.size() != other.data.size() || data[0].size() != other.data[0].size())
            throw std::invalid_argument("Matrix sizes do not match");
        Matrix<T> result(data.size(), data[0].size());
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[0].size(); ++j) {
                result.set(i, j, data[i][j] + other.get(i, j));
            }
        }
        return result;
    }
};

// -----------------------------
// Recursive Math Utilities
// -----------------------------
double fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

double power(double x, int n) {
    if (n == 0) return 1.0;
    if (n < 0) return 1.0 / power(x, -n);
    double half = power(x, n / 2);
    if (n % 2 == 0) return half * half;
    return x * half * half;
}

// -----------------------------
// Complex Logic Class
// -----------------------------
class ComplexProcessor {
    std::vector<int> data;
public:
    ComplexProcessor(size_t n) {
        for (size_t i = 0; i < n; ++i) data.push_back(rand() % 100);
    }

    int process() {
        int total = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] % 2 == 0) {
                total += factorial(data[i] % 10);
            } else if (data[i] % 3 == 0) {
                total += fibonacci(data[i] % 20);
            } else {
                total += power(2.0, data[i] % 10);
            }

            for (int j = 1; j <= 3; ++j) {
                if ((i + j) % 2 == 0) total -= j;
                else total += j;
            }
        }
        return total;
    }

    std::vector<int> filterEven() const {
        std::vector<int> result;
        for (auto v : data) if (v % 2 == 0) result.push_back(v);
        return result;
    }
};

// -----------------------------
// Main Function
// -----------------------------
int main() {
    Graph g;
    for (int i = 1; i <= 10; ++i) g.addNode(i);
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(2, 4);
    g.addEdge(2, 5);
    g.addEdge(3, 6);
    g.addEdge(3, 7);
    g.addEdge(4, 8);
    g.addEdge(5, 9);
    g.addEdge(6, 10);

    std::cout << "DFS sum from 1: " << g.sumValuesDFS(1) << "\n";

    Matrix<int> mat1(5, 5);
    Matrix<int> mat2(5, 5);
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) {
            mat1.set(i, j, i + j);
            mat2.set(i, j, i * j);
        }

    Matrix<int> mat3 = mat1 + mat2;
    std::cout << "Matrix[2][3] sum: " << mat3.get(2, 3) << "\n";

    ComplexProcessor processor(50);
    std::cout << "Complex processing result: " << processor.process() << "\n";

    auto evens = processor.filterEven();
    std::cout << "Filtered evens: ";
    for (auto v : evens) std::cout << v << " ";
    std::cout << "\n";

    std::map<std::string, int> wordCount;
    std::vector<std::string> words = {"apple", "banana", "apple", "cherry", "banana", "apple"};
    for (auto& w : words) wordCount[w]++;
    for (auto& [word, count] : wordCount)
        std::cout << word << ": " << count << "\n";

    return 0;
}
