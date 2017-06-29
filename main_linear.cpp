#include "preliminary.hpp"

int main() {
    vector<double> percentage;
	// record file for confusion matrix
    file::ofstream fout{root/"predict"/"confusion_linear.txt",ios::trunc};
	chrono::duration<double> time_train;
	chrono::duration<double> time_predict;
	for(auto round=1;round<=group_count;++round){
	    auto time_1=steady_clock::now();
        nnet unit;
        unit.init<problem>(round)
		    .init<parameter>(6)
		    .train();
		// record train time
		time_train+=(steady_clock::now()-time_1);
		auto time_2=steady_clock::now();
		auto result=unit.predict();
		// record predict time
		time_predict+=(steady_clock::now()-time_2);
        file::ifstream yfs{root/ytest(round)};
        assert(yfs);
        fout<<round<<en;
		// first line per group is desire value in record file
        copy(result.begin(),result.end(),ostream_iterator<double>(fout,"\t"));
        fout<<en;
        istream_iterator<string> iter(yfs);
        auto correct=count_if(result.begin(),result.end(),[&](int desire){
            auto line=*iter++;
            auto actual=lexical_cast<int>(line);
	        // second line per group is actual value  in record file
	        fout<<actual<<et;
            return equal_to<int>{}(desire,actual);
        });
        fout<<en;
        percentage.push_back(divides<double>()(correct,result.size()));
    }
	cout<<"train"<<et<<time_train<<en;
	cout<<"predict"<<et<<time_predict<<en;
    cout<<"accuracy"<<et<<percentage<<en;
    cout<<"average"<<et<<accumulate(percentage.begin(),percentage.end(),0.0)/percentage.size()<<en;
    return 0;
}