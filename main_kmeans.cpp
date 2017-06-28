#include "preliminary.hpp"

const auto group_count=45;
const auto tmp=5;
int main() {
	cout<<boolalpha;
	vector<double> percentage;
	file::ofstream fout{root/"predict"/"confusion_minmax.txt",ios::trunc};
	// model cluster
	vector<vector<unique_future<nnet>>> pool(group_count);
	assert(pool.size()==45);
	const auto module=4;
	// 4 origin input module, 2 min module with 1 max module
	// therefore origin index module number is between 0 and 3
	randomizer<int> seed{0,module-1};
	// train phase
	for(auto round=1;round<=tmp;++round){
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
			task.push_back(boost::async(boost::launch::async,[&,i,xptr=xs[i],yptr=ys[i]] {
				nnet unit;
				unit.init<problem>(round,*yptr,*xptr)
						.init<parameter>(6)
						.train();
				return unit;
			}));
		}
		assert(task.size()==4);
	}
	// call-back for all asynchronous train tasks
	for_each(pool.begin(),pool.end(),[&](vector<unique_future<nnet>>& vec){
		wait_for_all(vec.begin(),vec.end());
	});
	// predict phase
	for(auto round=1;round<=tmp;++round){
		file::ifstream xfin{root/xtest(round)};
		file::ifstream yfin{root/ytest(round)};
		assert(xfin&&yfin);
		fout<<round<<en;
		vector<string> xvec{istream_iterator<string>{xfin},istream_iterator<string>{}};
		istream_iterator<string> yiter{yfin};
		auto& task=pool[round-1];
		vector<vector<double>> result[module];
		once_flag token;
		vector<int> label;
		for(auto k=0;k<module;++k){
			auto model=task[k].get();
			call_once(token,[&]{
				auto ptr=model.modptr;
				label.assign(ptr->label,ptr->label+ptr->nr_class);
				assert(label.size()==3);
			});
			auto tmp=model.decision_value(xvec);
			result[k]=move(tmp);
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
		auto& decision=result[0];
		ostringstream oss;
		auto correct=count_if(decision.begin(),decision.end(),
		                      [&](vector<double>& vec)->bool{
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
	cout<<"per"<<et<<percentage<<en;
	cout<<"av"<<et<<accumulate(percentage.begin(),percentage.end(),0.0)/percentage.size()<<en;

	cout<<"finish"<<en;
	return 0;
}