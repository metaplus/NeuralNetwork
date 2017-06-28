#include "preliminary.hpp"

int main() {
    cout<<boolalpha;
    vector<double> percentage;
    file::ofstream fout{root/"predict"/"confusion_linear.txt",ios::trunc};
    for(auto round=1;round<=45;++round){
	    vector<nnet> cluster;
        nnet unit;
        auto result=unit
                .init<problem>(round)
                .init<parameter>(6)
                .train()
                .predict();
        file::ifstream yfs{root/ytest(round)};
        assert(yfs);
        fout<<round<<en;
        copy(result.begin(),result.end(),ostream_iterator<double>(fout,"\t"));
        fout<<en;
        istream_iterator<string> iter(yfs);
        auto correct=count_if(result.begin(),result.end(),[&](int desire){
            auto line=*iter++;
            auto actual=lexical_cast<int>(line);
            fout<<actual<<et;
            return desire==actual;
        });
        fout<<en;
        percentage.push_back(divides<double>()(correct,result.size()));
    }
    cout<<"per"<<et<<percentage<<en;
    cout<<"av"<<et<<accumulate(percentage.begin(),percentage.end(),0.0)/percentage.size()<<en;

    cout<<"finish"<<en;
    return 0;
}