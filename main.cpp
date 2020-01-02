//
// Created by Lukas Eßmann on 01/01/2020.
// Copyright (c) 2020 Lukas Eßmann. All rights reserved.
//

#include "src/System.h"
#include "src/Trainer.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

using namespace NanoID;

// ----------------------------------------------------------------------------------------

// We will need to create some functions for loading data.  This program will
// expect to be given a directory structured as follows:
//    top_level_directory/
//        person1/
//            image1.jpg
//            image2.jpg
//            image3.jpg
//        person2/
//            image4.jpg
//            image5.jpg
//            image6.jpg
//        person3/
//            image7.jpg
//            image8.jpg
//            image9.jpg
//
// The specific folder and image names don't matter, nor does the number of folders or
// images.  What does matter is that there is a top level folder, which contains
// subfolders, and each subfolder contains images of a single person.

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "Give a command as input. Possible commands are:" << endl;
        cout << "   train => Train a new ResNet" << endl;
        return 1;
    }

    if (std::string(argv[1]) == "train") {
        if (argc < 7)
        {
            cout << "ERROR: To few arguments. Exiting..." << endl;
            return 2;
        }
        auto t = new Trainer();
        // threshold should be 10000
        auto net = t->Train(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        t->SaveNet(net, argv[6]);
        return 0;
    } else {
        cout << "ERROR: " << argv[1] << " is not a valid command. Exiting..." << endl;
        return 2;
    }

    cout << "Detected " << System::GetCores() << " cpu cores." << endl;


    if (argc != 2)
    {
        cout << "Give a folder as input. It should contain sub-folders of images and we will " << endl;
        cout << "learn to distinguish between these sub-folders with metric learning. " << endl;
        cout << "For example, you can run this program on the very small examples/johns dataset " << endl;
        cout << "that comes with dlib by running this command:" << endl;
        cout << "   ./NanoID johns" << endl;
        return 1;
    }

//    // Now, just to show an example of how you would use the network, let's check how well
//    // it performs on the training data.
//    dlib::rand rnd(time(0));
//    load_mini_batch(5, 5, rnd, objs, images, labels);
//
//    // Normally you would use the non-batch-normalized version of the network to do
//    // testing, which is what we do here.
//    anet_type testing_net = net;
//
//    // Run all the images through the network to get their vector embeddings.
//    std::vector<matrix<float,0,1>> embedded = testing_net(images);
//
//    // Now, check if the embedding puts images with the same labels near each other and
//    // images with different labels far apart.
//    int num_right = 0;
//    int num_wrong = 0;
//    for (size_t i = 0; i < embedded.size(); ++i)
//    {
//        for (size_t j = i+1; j < embedded.size(); ++j)
//        {
//            if (labels[i] == labels[j])
//            {
//                // The loss_metric layer will cause images with the same label to be less
//                // than net.loss_details().get_distance_threshold() distance from each
//                // other.  So we can use that distance value as our testing threshold.
//                if (length(embedded[i]-embedded[j]) < testing_net.loss_details().get_distance_threshold())
//                    ++num_right;
//                else
//                    ++num_wrong;
//            }
//            else
//            {
//                if (length(embedded[i]-embedded[j]) >= testing_net.loss_details().get_distance_threshold())
//                    ++num_right;
//                else
//                    ++num_wrong;
//            }
//        }
//    }
//
//    cout << "num_right: "<< num_right << endl;
//    cout << "num_wrong: "<< num_wrong << endl;

}


