/*
 * exception.h
 *
 *  Created on: Feb 7, 2014
 *      Author: algomorph
 */

#ifndef EXCEPTION_HPP_
#define EXCEPTION_HPP_

#include <boost/exception/all.hpp>
#include <boost/throw_exception.hpp>

typedef boost::error_info<struct tag_error_message, const char*> error_message;
typedef boost::error_info<struct tag_cl_errror_code, int> cl_error_code;
struct runtime_error: virtual boost::exception, virtual std::exception {
};
struct logic_error: virtual boost::exception, virtual std::exception {
};



#endif /* EXCEPTION_HPP_ */
