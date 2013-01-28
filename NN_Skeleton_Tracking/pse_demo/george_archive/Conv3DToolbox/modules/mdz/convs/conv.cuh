#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

/*
 * This specifies the output order of the convolution routines in this file, or
 * the order of the images for conv2.
 */
enum ORDER {
    GROUP_FILTER_IMAGE, IMAGE_GROUP_FILTER
};
