#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <deque>
#include <stack>
#include <queue>
#include <utility>
#include <limits>
#include <random>

// -----------------------------
// Utility Functions and Templates
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

template<typename T>
T fibonacci(T n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

template<typename T>
T power(T base, int exp) {
    if (exp == 0) return 1;
    if (exp < 0) return 1 / power(base, -exp);
    T half = power(base, exp / 2);
    if (exp % 2 == 0) return half * half;
    return base * half * half;
}

// -----------------------------
// Graph Structure
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
        for (auto& [_, n] : nodes) delete n;
    }

    void addNode(int v) {
        if (nodes.find(v) == nodes.end()) nodes[v] = new Node(v);
    }

    void addEdge(int from, int to) {
        addNode(from);
        addNode(to);
        nodes[from]->children.push_back(nodes[to]);
    }

    int sumDFS(int start) {
        std::set<int> visited;
        return dfsHelper(nodes[start], visited);
    }

    int countNodesWithCondition(std::function<bool(int)> pred) {
        int count = 0;
        std::set<int> visited;
        for (auto& [k, node] : nodes) {
            count += countDFS(node, visited, pred);
        }
        return count;
    }

private:
    int dfsHelper(Node* node, std::set<int>& visited) {
        if (!node || visited.count(node->value)) return 0;
        visited.insert(node->value);
        int sum = node->value;
        for (auto child : node->children) sum += dfsHelper(child, visited);
        return sum;
    }

    int countDFS(Node* node, std::set<int>& visited, std::function<bool(int)> pred) {
        if (!node || visited.count(node->value)) return 0;
        visited.insert(node->value);
        int count = pred(node->value) ? 1 : 0;
        for (auto child : node->children) count += countDFS(child, visited, pred);
        return count;
    }
};

// -----------------------------
// Matrix Template
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
        for (size_t i = 0; i < data.size(); ++i)
            for (size_t j = 0; j < data[0].size(); ++j)
                result.set(i, j, data[i][j] + other.get(i, j));
        return result;
    }

    T sum() const {
        T s = 0;
        for (auto& row : data) s += std::accumulate(row.begin(), row.end(), T(0));
        return s;
    }
};

// -----------------------------
// Complex Processor
// -----------------------------
class Processor {
    std::vector<int> data;
public:
    Processor(size_t n) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(1, 50);
        for (size_t i = 0; i < n; ++i) data.push_back(dist(rng));
    }

    int compute() {
        int total = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] % 2 == 0) total += factorial(data[i] % 10);
            else if (data[i] % 3 == 0) total += fibonacci(data[i] % 20);
            else total += power(2, data[i] % 10);

            for (int j = 1; j <= 5; ++j) {
                if ((i + j) % 2 == 0) total -= j;
                else total += j;
                if (j % 3 == 0 && total > 1000) total /= 2;
            }

            if (data[i] % 7 == 0) {
                int innerSum = 0;
                for (int k = 1; k <= data[i] % 5 + 1; ++k)
                    innerSum += k * k;
                total += innerSum;
            }
        }
        return total;
    }

    std::vector<int> filterByModulo(int mod) const {
        std::vector<int> result;
        for (auto v : data) if (v % mod == 0) result.push_back(v);
        return result;
    }
};

// -----------------------------
// Advanced Recursive Algorithms
// -----------------------------
int ackermann(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ackermann(m - 1, 1);
    return ackermann(m - 1, ackermann(m, n - 1));
}

int collatzSteps(int n) {
    int steps = 0;
    while (n != 1) {
        if (n % 2 == 0) n /= 2;
        else n = 3 * n + 1;
        ++steps;
    }
    return steps;
}

// -----------------------------
// Main Function
// -----------------------------
int main() {
    Graph g;
    for (int i = 1; i <= 15; ++i) g.addNode(i);

    g.addEdge(1, 2); g.addEdge(1, 3); g.addEdge(2, 4); g.addEdge(2, 5);
    g.addEdge(3, 6); g.addEdge(3, 7); g.addEdge(4, 8); g.addEdge(5, 9);
    g.addEdge(6, 10); g.addEdge(7, 11); g.addEdge(8, 12); g.addEdge(9, 13);
    g.addEdge(10, 14); g.addEdge(11, 15);

    std::cout << "DFS sum from 1: " << g.sumDFS(1) << "\n";
    std::cout << "Count nodes divisible by 3: " << g.countNodesWithCondition([](int v){ return v % 3 == 0; }) << "\n";

    Matrix<int> mat1(10, 10), mat2(10, 10);
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j) {
            mat1.set(i,j, i+j);
            mat2.set(i,j, i*j);
        }

    Matrix<int> mat3 = mat1 + mat2;
    std::cout << "Matrix[5][5] sum: " << mat3.get(5,5) << ", total sum: " << mat3.sum() << "\n";

    Processor proc(100);
    std::cout << "Processor compute result: " << proc.compute() << "\n";

    auto multiplesOf5 = proc.filterByModulo(5);
    std::cout << "Multiples of 5: ";
    for (auto v : multiplesOf5) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Ackermann(3, 3): " << ackermann(3,3) << "\n";
    std::cout << "Collatz steps for 27: " << collatzSteps(27) << "\n";

    std::map<std::string,int> wordCount;
    std::vector<std::string> words = {"apple","banana","apple","cherry","banana","apple","date","fig","grape"};
    for (auto &w : words) wordCount[w]++;
    for (auto &[word,count] : wordCount) std::cout << word << ": " << count << "\n";

    // Complex nested loops for stress-testing cyclomatic complexity
    int complexSum = 0;
    for (int i = 1; i <= 20; ++i) {
        for (int j = 1; j <= 20; ++j) {
            for (int k = 1; k <= 5; ++k) {
                if ((i+j+k) % 2 == 0) complexSum += i*j*k;
                else if ((i*j - k) % 3 == 0) complexSum -= k;
                else if (i % 2 == 0 && j % 3 == 0) complexSum += k*k;
                else complexSum += i + j + k;

                switch (k % 4) {
                    case 0: complexSum += i; break;
                    case 1: complexSum -= j; break;
                    case 2: complexSum *= 1; break;
                    case 3: complexSum /= 1; break;
                }

                if (complexSum > 1000) complexSum %= 100;
            }
        }
    }
    std::cout << "Complex nested loop sum: " << complexSum << "\n";

    return 0;
}
