/*
 * test.h
 *
 *  Created on: Feb 17, 2014
 *      Author: algomorph
 */

#ifndef TEST_HPP_
#define TEST_HPP_

#include <cstdio>
#include <iostream>
#include <iomanip>

// Print a matrix of values
template<class T>
void prettyPrintMatrix(const T *img, const int& w, const int& h, const int& fw =
		4) {
	for (int i = 0; i < h; ++i) {
		std::cout << std::setw(fw) << img[i * w];
		for (int j = 1; j < w; ++j)
			std::cout << " " << std::setw(fw) << img[i * w + j];
		std::cout << "\n";
	}
}

#endif /* TEST_HPP_ */
