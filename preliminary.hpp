#pragma once
#include "common/cite.hpp"
#include "liblinear/linear.h"
#include "literal.hpp"

const file::path root{file::current_path().parent_path()/"dataset"};

struct base{
    problem prob;
    parameter para;
};

namespace impl{
    template<typename T,typename... Args>
    void initial(base&,Args...);
    template<>
    void initial<problem>(base& bs,int id){
        auto& problem=bs.prob;
        file::ifstream xfs{root/xtrain(id)};
        file::ifstream yfs{root/ytrain(id)};
        assert(xfs&&yfs);
        string line1;
        string line2;
        getline(xfs,line1);
        cout<<distance(istream_iterator<string>(yfs),istream_iterator<string>())<<en;
        //getline(yfs,line2);
        yfs.clear();
        yfs.seekg(ios::beg);
        cout<<distance(istream_iterator<string>(yfs),istream_iterator<string>())<<en;
        cout<<count(line1.rbegin(),line1.rend(),',')<<en;
        cout<<line2<<"]["<<en;
        move(istream_iterator<string>(yfs),istream_iterator<string>(),ostream_iterator<string>(cout,"\t"));

    };
    template<>
    void initial<parameter>(base& bs,int a){
        cout<<"fair ok"<<en;
        cout<<a<<en;
    };
}


class unit:public base{
public:
    template<typename U,typename... Args>
    void init(Args... a){
        impl::initial<U>(*this,forward<Args>(a)...);
    }
private:


};

