#include "preliminary.hpp"

int main() {
	// vector structure, statistics of accuracy
	vector<double> percentage;
	// record file of confusion matrix
	file::ofstream fout{root/"predict"/"confusion_minmax.txt",ios::trunc};
	// model cluster, as callback handler of asynchronous parallel tasks
	vector<vector<unique_future<nnet>>> pool(group_count);
	// 4 origin input module, 2 min module with 1 max module
	// therefore origin index module number is between 0 and 3
	randomizer<int> seed{0,module-1};
	// train phase
	chrono::duration<double> time_train;
	chrono::duration<double> time_predict;
	auto time_base=steady_clock::now();
	for(auto round=1;round<=group_count;++round){
		auto& task=pool[round-1];
		file::ifstream xfs{root/xtrain(round)};
		file::ifstream yfs{root/ytrain(round)};
		assert(xfs&&yfs);
		// select ptr facilitating efficient copy to asynchronous task
		vector<shared_ptr<vector<string>>> ys(module);
		vector<shared_ptr<vector<string>>> xs(module);
		for(auto i=0;i<module;++i){
			ys[i]=make_shared<vector<string>>();
			xs[i]=make_shared<vector<string>>();
		}
		string line1,line2;
		while(getline(yfs,line1)&&getline(xfs,line2)){
			auto index=seed();
			auto& y=*(ys[index]);
			auto& x=*(xs[index]);
			y.push_back(move(line1));
			x.push_back(move(line2));
		}
		for(auto i=0;i<module;++i){
			task.push_back(boost::async([&,i,xptr=xs[i],yptr=ys[i]] {
				nnet unit;
				unit.init<problem>(round,*yptr,*xptr)
					.init<parameter>(6)
					.train();
				return unit;
			}));
		}
	}
	// call-back for all asynchronous train tasks
	// this loop will block until all task finished and returned
	for_each(pool.begin(),pool.end(),[&](vector<unique_future<nnet>>& vec){
		wait_for_all(vec.begin(),vec.end());
	});
	time_train=steady_clock::now()-time_base;
	// predict phase
	for(auto round=1;round<=group_count;++round){
		auto time_mark=steady_clock::now();
		file::ifstream xfin{root/xtest(round)};
		file::ifstream yfin{root/ytest(round)};
		assert(xfin&&yfin);
		fout<<round<<en;
		vector<string> str{istream_iterator<string>{xfin},istream_iterator<string>{}};
		istream_iterator<string> yiter{yfin};
		auto& task=pool[round-1];
		vector<vector<double>> result[module];
		static const vector<int> label{1,0,-1};
		for(auto k=0;k<module;++k){
			// get module and decision value
			result[k]=task[k].get().decision_value(str);
		}
		// concise unified lambda expression for comparison
		auto merge=[&result](int i,int j,function<bool(double,double)> func){
			auto& v1=result[i];
			auto& v2=result[j];
			assert(v1.size()==v2.size());
			for(auto m=0;m<v1.size();++m){
				for(auto n=0;n<v1.back().size();++n){
					if(!func(v1[m][n],v2[m][n]))
						v1[m][n]=v2[m][n];
				}
			}
		};
		// min phase, target 0 and 2
		merge(0,1,less<double>{});
		merge(2,3,less<double>{});
		// max phase, target 0
		merge(0,2,greater<double>{});
		// make the final decision value;
		auto& decision=result[0];
		time_predict+=(steady_clock::now()-time_mark);
		ostringstream oss;
		auto correct=count_if(decision.begin(),decision.end(),[&](vector<double>& vec)->bool{
			auto line=*yiter++;
			auto actual=lexical_cast<int>(line);
			// vote for desired label
			auto offset=distance(vec.begin(),max_element(vec.begin(),vec.end()));
			fout<<label[offset]<<et;
			oss<<actual<<et;
			return label[offset]==actual;
		});
		fout<<en<<oss.str()<<en;
		percentage.push_back(divides<double>()(correct,decision.size()));
	}
	cout<<"train"<<et<<time_train<<en;
	cout<<"predict"<<et<<time_predict<<en;
	cout<<"accuracy"<<et<<percentage<<en;
	cout<<"average"<<et<<accumulate(percentage.begin(),percentage.end(),0.0)/percentage.size()<<en;
	return 0;
}
