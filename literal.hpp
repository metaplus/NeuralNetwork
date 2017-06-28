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

inline
vector<feature_node> parse_feature(const string& line){
    vector<feature_node> feature;
    auto index=0;
    auto pos=0;
    string value;
    while(line.find(',',pos)!=string::npos){
        auto next=line.find(',',pos);
        value=line.substr(pos,next-pos);
        feature.push_back(feature_node{++index,stod(value)});
        pos=next+1;
    }
    value=line.substr(pos);
    feature.push_back(feature_node{++index,stod(value)});
    feature.push_back(feature_node{-1,0});
    return feature;
}