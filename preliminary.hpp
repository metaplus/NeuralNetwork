#pragma once
#include "common/cite.hpp"
#include "liblinear/linear.h"
#include "literal.hpp"

// global configuration variable
const file::path root{file::current_path().parent_path()/"dataset"};
const auto group_count=45;
const auto module=4;

// base class for nnet class
struct base{
    int index;
    problem prob;
    parameter para;
    model* modptr;
    vector<double> label;
    vector<feature_node*> attr;
    vector<shared_ptr<vector<feature_node>>> manager;
};

// general template, its specialization is defined in the end
namespace impl{
    template<typename T,typename... Args>
    void initial(base&,Args...);
}

class nnet:public base{
public:
	// elegant visitor pattern in c++ design pattern
    template<typename U,typename... Args>
    nnet& init(Args... a){
		// all initial function will be forwarded to its specialization with varied parameters
		impl::initial<U>(*this,forward<Args>(a)...);
        return *this;
    }
    nnet& train(){
	    // check validation before training
        assert(check_parameter(&prob,&para)==nullptr);
	    // call global field lib-linear train function
        modptr=::train(&prob,&para);
        return *this;
    }
	// predict directly
    vector<double> predict(){
        vector<double> result;
        file::ifstream xfs{root/xtest(index)};
        assert(xfs);
		for_each(istream_iterator<string>{xfs},istream_iterator<string>{},
		         [&](const string& line){
					result.push_back(::predict(modptr,parse_feature(line)->data()));
		         });
        return result;
    }
	// return decision value before voting
	vector<vector<double>> decision_value(const vector<string>& test){
		vector<vector<double>> result(test.size());
		transform(test.cbegin(),test.cend(),result.begin(),[&,this](const string& line){
			vector<double> temp(modptr->nr_class);
			vector<double> decision(3);
			::predict_values(modptr,parse_feature(line)->data(),temp.data());
			if(temp.size()==3){
				return temp;
			}else{
				// in k means aggregation, it's possible for a model with less properties
				// therefore, it's necessary to mark 0 as neutral for vanish feature
				vector<int> label{modptr->label,modptr->label+modptr->nr_class};
				for(auto i=0;i<label.size();i++){
					switch(label[i]){
						case 1:decision[0]=temp[i];break;
						case 0:decision[1]=temp[i];break;
						case -1:decision[2]=temp[i];break;
					}
				}
				return decision;
			}
		});
		return result;
	}
};


// template specialization definition
namespace impl{
	template<>
	void initial<problem>(base& bs,int id,vector<string> y,vector<string> x){
		assert(y.size()==x.size());
		bs.attr.resize(y.size());
		transform(y.begin(),y.end(),x.begin(),bs.attr.begin(),
		          [&](const string& line1,const string& line2)->feature_node*{
			          bs.label.push_back(lexical_cast<double>(line1));
			          bs.manager.push_back(parse_feature(line2));
			          return bs.manager.back()->data();
		          });
		bs.prob.n=std::count(x.back().begin(),x.back().end(),',')+2;
		bs.prob.x=bs.attr.data();
		bs.prob.y=bs.label.data();
		bs.prob.l=y.size();
		bs.prob.bias=0;
		bs.index=id;
	}
	template<>
	void initial<problem>(base& bs,int id){
		file::ifstream xfs{root/xtrain(id)};
		file::ifstream yfs{root/ytrain(id)};
		assert(xfs&&yfs);
		string line1,line2;
		vector<string> y,x;
		while(getline(yfs,line1)&&getline(xfs,line2)){
			y.push_back(move(line1));
			x.push_back(move(line2));
		}
		initial<problem>(bs,id,move(y),move(x));
	};
	template<>
	void initial<parameter>(base& bs,int type){
		// default parameter
		auto& param=bs.para;
		param.solver_type = type;
		param.C = 1;
		param.eps = HUGE_VAL;
		param.p = 0.1;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		param.init_sol = NULL;
		if(param.eps == HUGE_VAL) {
			switch (param.solver_type) {
				case L2R_LR:
				case L2R_L2LOSS_SVC:param.eps = 0.01;break;
				case L2R_L2LOSS_SVR:param.eps = 0.001;break;
				case L2R_L2LOSS_SVC_DUAL:
				case L2R_L1LOSS_SVC_DUAL:
				case MCSVM_CS:
				case L2R_LR_DUAL:param.eps = 0.1;break;
				case L1R_L2LOSS_SVC:
				case L1R_LR:param.eps = 0.01;break;
				case L2R_L1LOSS_SVR_DUAL:
				case L2R_L2LOSS_SVR_DUAL:param.eps = 0.1;break;
			}
		}
	};
}

