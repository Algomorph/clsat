/*
 * test.h
 *
 *  Created on: Feb 17, 2014
 *      Author: algomorph
 */

#ifndef TEST_HPP_
#define TEST_HPP_

#include <cstdio>

template<class T>
void prettyPrintMat(T* matrix, const int& width, const int& height)
{
    int i, j;
    for (i = 0; i < height; ++i)
    {
        for (j = 0; j < width; ++j)
        	std::cout << matrix[i*width+j] << " ";
        printf("\n");
    }
}



#endif /* TEST_HPP_ */
