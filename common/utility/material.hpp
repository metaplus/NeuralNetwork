#pragma once
#include "common/citeSTL.hpp"
#include "common/citeBoost.hpp"
#include <cassert>
struct element
{
protected:
	ptree& tree;
	string index;
	file::path catalog;
	element(size_t id, ptree& tr, const file::path& catalog) :tree{ tr }, catalog{ catalog }
	{
		index = lexical_cast<string>(id);
	}
public:
	virtual void process(streambuf&, tcp::socket&, size_t) = 0;
};


struct json :element
{
	json(size_t id, ptree& tr, const file::path& catalog) :element{ id,tr,catalog }
	{
	}
	void process(streambuf& response, tcp::socket& socket, size_t length) override
	{
		BOOST_ASSERT(!length);
		cout <<en<< "here" << et << length << en;
		stringstream ss;
		string data{ istreambuf_iterator<char>(&response),istreambuf_iterator<char>{} };
		boost::regex_replace(std::ostreambuf_iterator<char>(ss), 
			data.begin(),data.end(),regex{ "[^{}]+\r\n" }, "");
		ptree node;
		property::read_json(ss, node);
		auto value = node.get<string>("plays");
		cout << "value" << et << index << et << value << en;
		tree.put(ptree::path_type{ index }/"Comments", value);
	}
};


struct mp3 :element
{

	mp3(size_t id, ptree& tr, const file::path& dir) :element{ id,tr,catalog }
	{
		static auto suffix = "mp3";
		catalog.append(suffix);
		static once_flag flag;
		call_once(flag,bind(file::create_directory,catalog));
		catalog.append(lexical_cast<string>(id)).replace_extension(suffix);
	}
	void process(streambuf& response, tcp::socket& socket, size_t length)
	{
		if (length > 10000000 || length < 800000)	return;
		cerr << "len" << et << length << en;
		auto ofptr = make_shared<file::ofstream>(catalog, ios::binary | ios::trunc);
		*ofptr << &response;
		async_read(socket, response, transfer_exactly(length),
			[&,ofptr](const error& ec, size_t transferred)
		{
			*ofptr << &response;
		});
	}
};


struct html :element
{

	html(size_t id, ptree& tr, const file::path& catalog) :element{ id,tr,catalog }
	{
	}
	void process(streambuf& response, tcp::socket& socket, size_t length) override
	{
		iostream html_stream{ &response };
		streambuf rest;
		asio::read_until(socket, rest, "0\r\n\r\n");
		html_stream << &rest;
		string page{ istreambuf_iterator<char>(&response),istreambuf_iterator<char>{} };
		cout <<et<< "db" << et <<index<<et<< page.size() << en;
		regex test{ "<meta property=\"og:music:artist\" content=\"Kyla\"\/>" };
		auto find_property=[&page](string key)
		{
			auto prev{ string{ "<meta property=\"og:" }+key + "\" content=\"" };
			auto pos1 = page.find(prev);
			auto pos2 = page.find("\"\/>",pos1);
			// amazingly, subjacent comment method doesn't work... maybe caused of encoding???
			//	cout << page.substr(pos1 + prev.size(), pos2-pos1) << en;    
			pos1 += prev.size();
			return pos2 >= page.size() ? string{} : string{ &page.at(pos1), &page.at(pos2) };
		};

		ptree node;
		node.put("SongName", find_property("title"));
		node.put("Artist", find_property("music:artist"));
		node.put("Album", find_property("music:album"));
		node.put("AlbumPic", find_property("image"));
		node.put("UrlProxy", string{ "http://link.hhtjim.com/xiami/" }+index + ".mp3");
		tree.add_child(index, node);
	}
};