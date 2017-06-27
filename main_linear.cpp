#include "preliminary.hpp"

int main() {
    cout<<boolalpha;



    nnet unit;

    unit.init<problem>(1);
    unit.init<parameter>(int{L2R_L2LOSS_SVC});
    auto model=unit.train();
    auto result=unit.predict();
    cout<<"finish"<<en;
    return 0;
}