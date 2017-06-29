#include "preliminary.hpp"


// node structure for k-means aggregation
struct spot{
	int label;
	vector<double> attr;
	string yline;
	string xline;
};

// euclidean distance calculation formula
template<typename T=double> inline
T euclidean2(const vector<T>& left,const vector<T>& right){
	T init=0;
	assert(left.size()==right.size());
	for(auto i=0;i<left.size();++i){
		init+=pow<T>(left[i]-right[i],2);
	}
	return init;
	// square root is redundant, waste 1 second in total 45 models
	// return sqrt(init);
}

// equal to vector add operation
template<typename T=double> inline
vector<T>& operator+=(vector<T>& left,const vector<T>& right){
	assert(left.size()==right.size());
	transform(left.begin(),left.end(),right.begin(),
	          left.begin(),plus<T>{});
	return left;
}

// equal to vector divide operation
// meta-programming assures that U type can safely convert to T type
template<typename T=double,typename U,typename=enable_if_t<is_same<common_type_t<T,U>,T>::value>> inline
vector<T>& operator/=(vector<T>& left,const U& den){
	for(auto& val:left){
		val=divides<T>{}(val,den);
	}
	return left;
}

// randomly choose k different index number
// while fairly partition and present randomness in sub range may be better
inline
set<int> random_select(int low,int high,int num){
	static randomizer<int> seed{low,high};
	if(num==0||high-low+1<num)
		return {};
	else if(num==1){
		set<int> s;
		s.insert(seed());
		return s;
	}else{
		auto s=random_select(low,high,num-1);
		auto old=s.size();
		while(old==s.size()){
			s.insert(seed());
		}
		return s;
	}
}


int main() {
	vector<double> percentage;
	file::ofstream fout{root/"predict"/"confusion_kmeans.txt",ios::trunc};
	// model cluster
	vector<vector<unique_future<nnet>>> pool(group_count);
	assert(pool.size()==45);
	const auto module=4;
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
		string line1,line2;
		vector<spot> all;  // cluster beyond centers
		while(getline(yfs,line1)&&getline(xfs,line2)){
			auto attribute=split_line<double>(line2);
			all.push_back(spot{lexical_cast<int>(line1),move(attribute),
			               move(line1),move(line2)});
		}
		// k means method implementation where choosing k = 4
		// firstly choose k appropriate initial center
		using iter_type=vector<spot>::iterator;
		vector<vector<iter_type>> cluster(module);
		auto center=random_select(0,all.size()-1,module);
		for(auto iter=all.begin();iter!=all.end();++iter){
			auto _it=min_element(center.begin(),center.end(),[&](int l,int r)->bool{
				return euclidean2(iter->attr,all[l].attr)<euclidean2(iter->attr,all[r].attr);
			});
			cluster[distance(center.begin(),_it)].push_back(iter);
		}
		// recursion k means call until convergence
		function<void(double)> k_means;
		// definition of the k means algorithm
		k_means=[&](double old=0)->void{
			// calculate new center vector
			vector<vector<double>> average;
			for(auto& vec:cluster){
				vector<double> sum=(*(vec.begin()))->attr;
				for(auto iter=vec.begin()+1;iter!=vec.end();++iter){
					auto& attr=(*iter)->attr;
					sum+=attr;
				}
				sum/=vec.size();
				average.push_back(move(sum));
				vec.resize(0);
			}
			// adjust to new clusters
			double entropy=0;
			for(auto iter=all.begin();iter!=all.end();++iter){
				auto _it=min_element(average.begin(),average.end(),[&](vector<double>& l,vector<double>& r)->bool{
					return euclidean2(iter->attr,l)<euclidean2(iter->attr,r);
				});
				entropy+=euclidean2(iter->attr,*_it);
				//entropy+=pow(euclidean(iter->attr,*_it),2);
				cluster[distance(average.begin(),_it)].push_back(iter);
			}
			// if condition satisfied, recursion happens for smaller entropy
			if(old==0||entropy<old)
				k_means(entropy);
		};
		// start k means aggregation procedure
		k_means(0);
		// generate legal input from iterator of cluster
		// iterator is a slight approach to copy object's reference better than raw pointer
	    for(auto& vec:cluster){
		    auto xptr=make_shared<vector<string>>();
		    auto yptr=make_shared<vector<string>>();
		    for(auto& iter:vec){
			    xptr->push_back(move(iter->xline));
			    yptr->push_back(move(iter->yline));
		    }
		    // async launch policy is default without specification
		    // therefore c++ runtime library will schedule how many threads need to be spawned with optimization
		    // it's been tested that default entrusted method performs better in large scale parallelism
		    // force on conventional 1-thread-1-task will leads to expensive thread launch and context switching
		    task.push_back(boost::async([&,xptr,yptr] {
			    nnet unit;
			    unit.init<problem>(round,*yptr,*xptr)
					    .init<parameter>(6)
					    .train();
			    return unit;
		    }));
		}
	}
	// call-back for all asynchronous train tasks
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
		vector<string> xvec{istream_iterator<string>{xfin},istream_iterator<string>{}};
		istream_iterator<string> yiter{yfin};
		auto& task=pool[round-1];
		vector<vector<double>> result[module];
	//	vector<int> label;
		for(auto k=0;k<module;++k){
			auto model=task[k].get();
			result[k]=model.decision_value(xvec);
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
		// draw final decision value before checking
		// predict phase is considered finished here
		auto& decision=result[0];
		time_predict+=(steady_clock::now()-time_mark);
		ostringstream oss;
		vector<int> label{1,0,-1};
		// check phase
		auto correct=count_if(decision.begin(),decision.end(),[&](vector<double>& vec)->bool{
			auto line=*yiter++;
			auto actual=lexical_cast<int>(line);
			// vote for desired label
			auto offset=distance(vec.begin(),max_element(vec.begin(),vec.end()));
			fout<<label[offset]<<et;
			oss<<actual<<et;
			return label[offset]==actual;
		});
		// record actual label temporarily contained in string stream structure
		// write back to log file under dataset/predict folder before making confusion matrix
		fout<<en<<oss.str()<<en;
		percentage.push_back(divides<double>()(correct,decision.size()));
	}
	cout<<"train"<<et<<time_train<<en;
	cout<<"predict"<<et<<time_predict<<en;
	cout<<"accuracy"<<et<<percentage<<en;
	cout<<"average"<<et<<accumulate(percentage.begin(),percentage.end(),0.0)/percentage.size()<<en;
	return 0;
}