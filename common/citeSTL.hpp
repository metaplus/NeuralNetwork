#pragma once

#include <iostream>
#include <algorithm>
#include <numeric>
#include <utility>
#include <valarray>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <set>
#include <fstream>
#include <functional>
#include <chrono>
#include <sstream>
#include <thread>
#include <limits>
#include <memory>
#include <future>
#include <random>
#include <queue>
#include <iterator>
#include <atomic>
#include <ratio>
#include <exception>
#include <cassert>

//namespace chrono = std::chrono;
//namespace place = std::placeholders;

using std::string;
using std::vector;
using std::set;
using std::map;
using std::unordered_map;
using std::ios;
using std::istringstream;
using std::ifstream;
using std::istream;
using std::ostringstream;
using std::ofstream;
using std::ostream;
using std::iostream;
using std::istream_iterator;
using std::ostream_iterator;
using std::istreambuf_iterator;
using std::ostreambuf_iterator;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::ratio;
using std::ratio_equal;
using std::ratio_greater;
using std::ratio_less;
using std::ratio_add;
using std::ratio_subtract;
using std::ratio_multiply;
using std::ratio_divide;
using std::numeric_limits;
using std::valarray;
using std::function;
using std::pair;
using std::tuple;
using std::shared_ptr;
using std::unique_ptr;
using std::enable_shared_from_this;
using std::stringstream;
using std::default_random_engine;
using std::enable_if;
using std::enable_if_t;
using std::conditional;
using std::is_same;
using std::is_base_of;
using std::is_function;
using std::once_flag;
using std::divides;
using std::multiplies;
using std::minus;
using std::plus;

using std::to_string;
using std::max;
using std::min;
using std::advance;
using std::distance;
using std::ref;
using std::move;
using std::forward;
using std::copy;
using std::accumulate;
using std::transform;
using std::count;
using std::count_if;
//using std::bind;
//using std::thread;
//using std::future;
//using std::promise;
using std::getline;
using std::ws;
using std::make_shared;
using std::make_unique;		// required compiler enable c++ 14 features
using std::call_once;
using std::boolalpha;
using std::stoi;
using std::stod;
using std::stol;
using std::stoul;

using std::cout;
using std::cin;
using std::cerr;
using std::clog;

enum symbol:char{
    en='\n',et='\t',er='\r',sp=' ',dt=','
};

inline
string operator""_str (const char* c){
    return string{c};
}

inline
ostream& crlf(ostream& os){
    return  os<<'\r'<<'\n';
}

template <typename T,typename Di=typename conditional<numeric_limits<T>().is_integer,
        uniform_int_distribution<>,uniform_real_distribution<>>::type>
class randomizer{
public:
    using value_type =T;
    using distribution_type=Di;
    randomizer(value_type low,value_type high){
        rd=bind(distribution_type{low,high},default_random_engine{});
    }
    value_type operator()(){
        return rd();
    }
private:
    function<value_type()> rd;
};

inline
ostream& operator<<(ostream& os,const symbol e){
    return os<<char{e};
}

template<typename T,typename U>
ostream& operator<<(ostream& os,pair<T,U> pr){
    return os<<pr.first<<' '<<pr.second<<en;
}
template <typename T>
ostream& operator<<(ostream& os,vector<T> vec){
    copy(vec.begin(),vec.end(),ostream_iterator<T>(cout,"\t"));
    return os;
}

inline
double sigmoid(double t){
    return 1/(1+exp(-t));
}

inline
double derived_sigmoid(double t){
    auto tmp=sigmoid(t);
    return tmp*(1-tmp);
}

inline
int hardlim(double d,double threshold=0.5){
    return d>threshold?1:0;
}

#ifdef WIN64
template <int a,int b> constexpr
bool equal()
{
	return std::ratio_equal_v<ratio<a>, ratio<b>>;

}
#endif
