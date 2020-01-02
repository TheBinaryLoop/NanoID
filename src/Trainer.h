//
// Created by Lukas Eßmann on 02/01/2020.
// Copyright (c) 2020 Lukas Eßmann. All rights reserved.
//

#ifndef NANOID_TRAINER_H
#define NANOID_TRAINER_H

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include "System.h"
#include "../includes/types.h"

using std::string;
using std::vector;
using dlib::matrix;
using dlib::rgb_pixel;
using dlib::directory;
using dlib::loss_metric;
using dlib::avg_pool_everything;
using dlib::input_rgb_image;
using dlib::max_pool;
using dlib::fc_no_bias;


// ----------------------------------------------------------------------------------------

// The next page of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and make the network somewhat smaller.

// training network type
using net_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                                            level0<
                                                    level1<
                                                            level2<
                                                                    level3<
                                                                            level4<
                                                                                    max_pool<3,3,2,2,relu<bn_con<con<32,7,7,2,2,
                                                                                    input_rgb_image
                                                                            >>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                                             alevel0<
                                                     alevel1<
                                                             alevel2<
                                                                     alevel3<
                                                                             alevel4<
                                                                                     max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                                                                     input_rgb_image
                                                                             >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

namespace NanoID {
    class Trainer {
    private:
        vector<vector<string>> LoadObjectsList (const string& dir);
        static void LoadMiniBatch (
                const size_t num_people,     // how many different people to include
                const size_t samples_per_id, // how many images per person to select.
                dlib::rand& rnd,
                const vector<vector<string>>& objs,
                vector<matrix<rgb_pixel>>& images,
                vector<unsigned long>& labels
        );
    public:
        net_type Train(const string &dir, const size_t numPeople, const size_t samplesPerId, unsigned long threshold);
        void SaveNet(net_type &net, const string &filename);
    };
}

#endif //NANOID_TRAINER_H
