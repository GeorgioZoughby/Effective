#include <iostream>
#include "../include/Vector.h"
#include "../include/VectorMul.h"
#include "../include/VectorAdd.h"
#include "../include/VectorSub.h"


int main() {
using namespace expr;
    Vector<int> v;

    std::cout << "Pushing 3 elements...\n";
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);

    std::cout << "Front: " << v.front() << "\n";
    std::cout << "Back: " << v.back() << "\n";
    std::cout << "At(1): " << v.at(1) << "\n";

    std::cout << "Using operator[]: ";
    for (int i = 0; i < v.size(); ++i) std::cout << v[i] << " ";
    std::cout << "\n";

    std::cout << "Size: " << v.size() << ", Capacity: " << v.capacity() << "\n";

    std::cout << "Pop back...\n";
    v.pop_back();
    std::cout << "Size: " << v.size() << ", Back: " << v.back() << "\n";

    std::cout << "Resizing to 5 with default value 99...\n";
    v.resize(5, 99);
    for (int i = 0; i < v.size(); ++i) std::cout << v[i] << " ";
    std::cout << "\n";

    std::cout << "Clearing vector...\n";
    v.clear();
    std::cout << "Size after clear: " << v.size() << ", Empty? " << v.empty() << "\n";

    std::cout << "Testing iterators (push then iterate):\n";
    for (int i = 1; i <= 5; ++i) v.push_back(i * 10);
    for (auto it = v.begin(); it != v.end(); ++it)
        std::cout << *it << " ";
    std::cout << "\n";

    Vector<int> v1;
    for (int i = 0; i < 5; ++i) v1.push_back(i + 1);

    // Arithmetic expressions
    Vector<int> sum = v + v1;
    Vector<int> diff = v - v1;
    Vector<int> prod = v * v1;
    Vector<int> scaled = v * 2;
    Vector<int> shifted = v + 5;

    std::cout << "v + v1: ";
    for (int i = 0; i < sum.size(); ++i) std::cout << sum[i] << " ";
    std::cout << "\nv - v1: ";
    for (int i = 0; i < diff.size(); ++i) std::cout << diff[i] << " ";
    std::cout << "\nv * v1: ";
    for (int i = 0; i < prod.size(); ++i) std::cout << prod[i] << " ";
    std::cout << "\nv * 2: ";
    for (int i = 0; i < scaled.size(); ++i) std::cout << scaled[i] << " ";
    std::cout << "\nv + 5: ";
    for (int i = 0; i < shifted.size(); ++i) std::cout << shifted[i] << " ";
    std::cout << "\n";
    std::cout << "v + 5: size = " << shifted.size() << "\n";


    std::cout << "Dot product v . v1: " << v.dot(v1) << "\n";

    std::cout << "Equality test: v == v1? " << (v == v1 ? "true" : "false") << "\n";
    std::cout << "Inequality test: v != v1? " << (v != v1 ? "true" : "false") << "\n";


    return 0;
}

