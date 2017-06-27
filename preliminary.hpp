#pragma once
#include "common/cite.hpp"
#include "liblinear/linear.h"
#include "literal.hpp"

const file::path root{file::current_path().parent_path()/"dataset"};
const file::path model_dir{file::current_path().parent_path()/"model"};

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

    template<>
    void initial<problem>(base& bs,int id){
        auto& problem=bs.prob;
        bs.index=id;
        file::ifstream xfs{root/xtrain(id)};
        file::ifstream yfs{root/ytrain(id)};
        assert(xfs&&yfs);
        string line1,line2;
        auto count=0;
        once_flag token;
        while(getline(yfs,line1)&&getline(xfs,line2)&&++count){
            bs.label.push_back(stod(line1));
            call_once(token,[&]{
                problem.n=std::count(line2.begin(),line2.end(),',')+2;
            });
            vector<feature_node> node;
            auto index=0;
            auto pos=3;
            string value;
            while(line2.find(',',pos)!=string::npos){
                auto next=line2.find(',',pos);
                value=line2.substr(pos,next-pos);
                node.push_back(feature_node{++index,stod(value)});
                pos=next+1;
            }
            value=line2.substr(pos);
            // lexical_cast surprisingly takes '\0' as error while stod doesn'
            node.push_back(feature_node{++index,stod(value)});
            node.push_back(feature_node{-1,0});
            assert(index==310);
            assert(node.size()==311);
            bs.temp.push_back(move(node));
            bs.attr.push_back(bs.temp.back().data());
            assert(bs.temp.back().size()==311);
        }
        problem.x=bs.attr.data();
        problem.y=bs.label.data();
        problem.l=count;
        problem.bias=0;
        cout<<problem.n<<en;
        cout<<"a"<<et<<bs.attr.size()<<en;
        cout<<"c"<<et<<count<<en;
    };
    template<>
    void initial<parameter>(base& bs,int type){
       // cout<<"t"<<et<<(type==L2R_L2LOSS_SVC_DUAL)<<en;
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


class nnet:public base{
public:
    template<typename U,typename... Args>
    void init(Args... a){   // elegant visitor design pattern
        impl::initial<U>(*this,forward<Args>(a)...);
    }
    model* train(){
        assert(check_parameter(&prob,&para)==nullptr);
        modptr=::train(&prob,&para);
        return modptr;
    }
    vector<double> predict(){
        vector<double> result;

        return {};
    }
private:


};

