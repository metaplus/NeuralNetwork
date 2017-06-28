#pragma once
#include "common/cite.hpp"
#include "liblinear/linear.h"
#include "literal.hpp"

const file::path root{file::current_path().parent_path()/"dataset"};

struct base{
    int index;
    problem prob;
    parameter para;
    model* modptr;
    vector<double> label;
    vector<feature_node*> attr;
    vector<vector<feature_node>> temp;
};

namespace impl{
    template<typename T,typename... Args>
    void initial(base&,Args...);
}


class nnet:public base{
public:
	// elegant visitor design pattern
    template<typename U,typename... Args>
    nnet& init(Args... a){
        impl::initial<U>(*this,forward<Args>(a)...);
        return *this;
    }
    nnet& train(){
        assert(check_parameter(&prob,&para)==nullptr);
        modptr=::train(&prob,&para);
	    cerr<<modptr->nr_class<<et<<modptr->nr_feature<<en;
        return *this;
    }
    vector<double> predict(){
        vector<double> result;
        file::ifstream xfs{root/xtest(index)};
        assert(xfs);
        string line;
        while(getline(xfs,line)){
            auto node=parse_feature(line);
	        result.push_back(::predict(modptr,node.data()));
        }
        return result;
    }
	vector<vector<double>> decision_value(const vector<string>& test){
		vector<vector<double>> result;
		transform(test.cbegin(),test.cend(),back_inserter(result),[&,this](const string& line){
			//vector<double> decision(modptr->nr_class);
			vector<double> tmp(modptr->nr_class);
			vector<double> decision(3);
			assert(decision[0]==0&&decision[1]==0&&decision[2]==0);
			auto node=parse_feature(line);
			::predict_values(modptr,node.data(),tmp.data());
			if(tmp.size()==3){
				decision=move(tmp);
			}else{
				vector<int> label{modptr->label,modptr->label+modptr->nr_class};
				assert(label.size()==tmp.size());
				for(auto i=0;i<label.size();i++){
					switch(label[i]){
						case 1:
							decision[0]=tmp[i];
							break;
						case 0:
							decision[1]=tmp[i];
							break;
						case -1:
							decision[2]=tmp[i];
							break;
						default:
							break;
					}
				}
			}
			return decision;
		});
		assert(result.size()==test.size());
		return result;
	}
};


// template specialization definition
namespace impl{
	template<>
	void initial<problem>(base& bs,int id,vector<string> y,vector<string> x){
		assert(y.size()==x.size());
		auto& problem=bs.prob;
		bs.index=id;
		once_flag token;
		auto cnt=0;
		transform(y.begin(),y.end(),x.begin(),back_inserter(bs.attr),
		          [&](string& line1,string& line2)->feature_node*{
			          ++cnt;
			          bs.label.push_back(lexical_cast<double>(line1));
			          call_once(token,[&]{
				          problem.n=std::count(line2.begin(),line2.end(),',')+2;
			          });
			          auto node=parse_feature(line2);
			          bs.temp.push_back(move(node));
			          return bs.temp.back().data();
		          });
		cerr<<"count"<<et<<cnt<<en;
		problem.x=bs.attr.data();
		problem.y=bs.label.data();
		problem.l=cnt;
		problem.bias=0;
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
				case L2R_L2LOSS_SVC:
					param.eps = 0.01;
					break;
				case L2R_L2LOSS_SVR:
					param.eps = 0.001;
					break;
				case L2R_L2LOSS_SVC_DUAL:
				case L2R_L1LOSS_SVC_DUAL:
				case MCSVM_CS:
				case L2R_LR_DUAL:
					param.eps = 0.1;
					break;
				case L1R_L2LOSS_SVC:
				case L1R_LR:
					param.eps = 0.01;
					break;
				case L2R_L1LOSS_SVR_DUAL:
				case L2R_L2LOSS_SVR_DUAL:
					param.eps = 0.1;
					break;
			}
		}
	};
}

