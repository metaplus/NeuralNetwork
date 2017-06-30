// the last piece of source code aiming to generate confusion matrix
// simple snippet, love holiday!
#include "preliminary.hpp"
#include "literal.hpp"

const file::path kms{root/"predict"/"confusion_kmeans.txt"};
const file::path lin{root/"predict"/"confusion_linear.txt"};
const file::path mmx{root/"predict"/"confusion_minmax.txt"};

int main(){
	file::ifstream kmeans{kms};
	file::ifstream linear{lin};
	file::ifstream minmax{mmx};
	assert(kmeans&&linear&&minmax);
	auto confusion_matrix=[&](file::ifstream& fin){
		string group_id;
		map<int,vector<int>> table{
				{1,vector<int>(3)},{0,vector<int>(3)},{-1,vector<int>(3)}
		};
		while(getline(fin,group_id)&&!group_id.empty()){
			string des,act;
			getline(fin,des);
			getline(fin,act);
			auto desire=split_line<int>(des,'\t');
			auto actual=split_line<int>(act,'\t');
			for(auto i=0;i<desire.size();++i){
				++table[actual[i]].at(1-desire[i]);
			}
		}
		cout<<table[1]<<en;
		cout<<table[0]<<en;
		cout<<table[-1]<<en;
	};
	// 3 confusion matrix will direct display in the terminal
	cout<<"linear"<<en;
	confusion_matrix(linear);
	cout<<"minmax"<<en;
	confusion_matrix(minmax);
	cout<<"kmeans"<<en;
	confusion_matrix(kmeans);

	return 0;
}