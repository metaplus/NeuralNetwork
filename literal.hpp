#pragma once

const string xtrain(int id){
    return string{"x_train_"}+lexical_cast<string>(id)+".csv";
}
const string xtest(int id){
    return string{"x_test_"}+lexical_cast<string>(id)+".csv";
}

const string ytrain(int id){
    return string{"y_train_"}+lexical_cast<string>(id)+".csv";
}
const string ytest(int id){
    return string{"y_test_"}+lexical_cast<string>(id)+".csv";
}