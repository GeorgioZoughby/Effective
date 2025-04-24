#include "../include/HMatrix.h"

int main() {
    Matrix<float> A(2, 2);
    Matrix<float> B(2, 2);
    A.setConstant(1.0f);
    B.setConstant(2.0f);
    Matrix<float> C = A + B; 
    std::cout << C;

    std::cout << "A == B? " << (A == B) << "\n";
    std::cout << "A != B? " << (A != B) << "\n";
    std::cout << "(A + B) == C? " << ((A + B) == C) << "\n";

    Matrix<float> D = A - B;
    std::cout << D;

    Matrix<float> M = A * 3.0f;
    std::cout << M;

    Matrix<float> N = 0.5f * A;
    std::cout<<N;

    Matrix<float> L = (A - B) / 2.0f;
    std::cout<<L;

    Matrix<float> M1(2, 3);
    Matrix<float> M2(3, 2);
    M1.setConstant(2.0f);
    M2.setConstant(1.0f);

    Matrix<float> M3 = M1 * M2;
    std::cout <<"M1 * M2 = \n"<< M3;

    std::cout<<"-M3 = \n"<< -M3;

    std::cout<<"M1 before transpose:\n"<<M1;

    std::cout<<"Transpose:\n"<<M1.transpose();

    Matrix<float> M4(2,2) , M5(2,2);
    M4.setConstant(2.0f);
    M5.setConstant(3.0f);

    std::cout << "Matrix M4:\n" << M4;
    std::cout << "Matrix M5:\n" << M5;

    // Element-wise multiplication
    Matrix<float> M6 = M4.cwiseProduct(M5);
    std::cout << "\nM4.cwiseProduct(M5):\n" << M6;

    // Matrix multiplication (dot product style)
    Matrix<float> M7 = M4 * M5;
    std::cout << "\nM4 * M5:\n" << M7;

    Matrix<float> M8 = M4.cwiseQuotient(M5);
    std::cout << "\nM4.cwiseQuotient(M5):\n" << M8;

    float result = M4.dot(M5);  
    std::cout << "M4.dot(M5) = " << result << '\n';

    std::cout<<"\nSum of M4\n"<<M4.sum()<<'\n';

    Matrix<float> A1(6,6);
    A1.setRandom();
    A1(0,0) = 6;
    std::cout<<"Random Matrix A1:\n"<<A1;

    std::cout << "A1 Min: " << A1.minCoeff() << "\n";   
    std::cout << "A1 Max: " << A1.maxCoeff() << "\n";


    Matrix<float> A2(8, 8);
    A2.setConstant(3.0f);
    std::cout << "A2:\n" << A2;
    std::cout << "Norm of A2: " << A2.norm() << "\n";

    Matrix<float> A3 = A2.normalized();
    std::cout << "Normalized A2:\n" << A3;
    std::cout << "Norm of A3: " << A3.norm();

    std::cout << "\nSecond row of A3:\n" << A3.row(1);
    std::cout << "\nSecond col of A3:\n" << A3.col(1);

    std::cout << "\nBlock (1,1) of size 2x2:\n" << A1.block(1, 1, 2, 2);

    return 0;
}
