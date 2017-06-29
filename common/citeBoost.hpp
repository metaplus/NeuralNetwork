#pragma once

#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/strand.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/unordered_map.hpp>
#include <boost/thread/future.hpp>
#include <boost/assert.hpp>
#include <boost/regex.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/asio/use_future.hpp>
#include <boost/chrono.hpp>
#include <boost/chrono/duration.hpp>

namespace asio=boost::asio;
namespace file=boost::filesystem;
namespace this_thread = boost::this_thread;
namespace lockfree = boost::lockfree;
namespace place = boost::asio::placeholders;
namespace chrono = boost::chrono;
namespace property = boost::property_tree;
namespace posix = boost::posix_time;


using error = boost::system::error_code;

using boost::barrier;
using boost::thread;
using boost::thread_group;
using boost::unique_future;
using boost::shared_future;
using boost::regex;
using boost::smatch;
using boost::cmatch;
using boost::promise;
using boost::optional;
using boost::packaged_task;
using boost::lockfree::spsc_queue;
using boost::lockfree::capacity;
using boost::asio::ip::address;
using boost::asio::ip::tcp;
using boost::asio::ip::udp;
using boost::asio::io_service;
using boost::asio::streambuf;
using boost::asio::use_future;
using boost::noncopyable;
using boost::filesystem::directory_iterator;
using boost::property_tree::ptree;
using boost::chrono::steady_clock;
using boost::wait_for_all;
using boost::wait_for_any;
using boost::launch;

using boost::bind;
using boost::asio::async_read_until;
using boost::asio::async_read;
using boost::asio::async_write;
using boost::asio::async_connect;
using boost::asio::read_until;
using boost::asio::read;
using boost::asio::write;
using boost::asio::connect;
using boost::regex_match;
using boost::regex_search;
using boost::lexical_cast;
using boost::asio::transfer_all;
using boost::asio::transfer_at_least;
using boost::asio::transfer_exactly;
