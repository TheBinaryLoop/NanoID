//
// Created by Lukas Eßmann on 02/01/2020.
// Copyright (c) 2020 Lukas Eßmann. All rights reserved.
//

#include "Trainer.h"

// This function spiders the top level directory and obtains a list of all the
// image files.
vector<vector<string>> NanoID::Trainer::LoadObjectsList(const string &dir) {
    vector<vector<string>> objects;
    for (auto subdir : directory(dir).get_dirs())
    {
        vector<string> imgs;
        for (auto img : subdir.get_files())
            imgs.push_back(img);

        if (imgs.size() != 0)
            objects.push_back(imgs);
    }
    return objects;
}

// This function takes the output of load_objects_list() as input and randomly
// selects images for training.  It should also be pointed out that it's really
// important that each mini-batch contain multiple images of each person.  This
// is because the metric learning algorithm needs to consider pairs of images
// that should be close (i.e. images of the same person) as well as pairs of
// images that should be far apart (i.e. images of different people) during each
// training step.
void NanoID::Trainer::LoadMiniBatch(const size_t num_people, const size_t samples_per_id, dlib::rand &rnd,
                                    const vector<vector<string>> &objs,
                                    vector<matrix<rgb_pixel>> &images,
                                    vector<unsigned long> &labels) {
    images.clear();
    labels.clear();
    DLIB_CASSERT(num_people <= objs.size(), "The dataset doesn't have that many people in it.");

    vector<bool> already_selected(objs.size(), false);
    matrix<rgb_pixel> image;
    for (size_t i = 0; i < num_people; ++i)
    {
        size_t id = rnd.get_random_32bit_number()%objs.size();
        // don't pick a person we already added to the mini-batch
        while(already_selected[id])
            id = rnd.get_random_32bit_number()%objs.size();
        already_selected[id] = true;

        for (size_t j = 0; j < samples_per_id; ++j)
        {
            const auto& obj = objs[id][rnd.get_random_32bit_number()%objs[id].size()];
            load_image(image, obj);
            images.push_back(std::move(image));
            labels.push_back(id);
        }
    }

    // You might want to do some data augmentation at this point.  Here we do some simple
    // color augmentation.
    for (auto&& crop : images)
    {
        disturb_colors(crop,rnd);
        // Jitter most crops
        if (rnd.get_random_double() > 0.1)
            crop = jitter_image(crop,rnd);
    }


    // All the images going into a mini-batch have to be the same size.  And really, all
    // the images in your entire training dataset should be the same size for what we are
    // doing to make the most sense.
    DLIB_CASSERT(images.size() > 0);
    for (auto&& img : images)
    {
        DLIB_CASSERT(img.nr() == images[0].nr() && img.nc() == images[0].nc(),
                     "All the images in a single mini-batch must be the same size.");
    }
}

net_type NanoID::Trainer::Train(const string &dir, const size_t numPeople, const size_t samplesPerId,
        unsigned long threshold) {

    auto objs = LoadObjectsList(dir);

    std::cout << "Label count: "<< objs.size() << std::endl;

    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;

    net_type net;

    dlib::dnn_trainer<net_type> trainer(net, dlib::sgd(0.0001, 0.9));

    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("face_metric_sync", std::chrono::minutes(5));
    // I've set this to something really small to make the example terminate
    // sooner. But when you really want to train a good model you should set
    // this to something like 10000 so training doesn't terminate too early.
    trainer.set_iterations_without_progress_threshold(threshold);

    // If you have a lot of data then it might not be reasonable to load it all
    // into RAM. So you will need to be sure you are decompressing your images
    // and loading them fast enough to keep the GPU occupied. I like to do this
    // using the following coding pattern: create a bunch of threads that dump
    // mini-batches into dlib::pipes.
    dlib::pipe<std::vector<matrix<rgb_pixel>>> qimages(4);
    dlib::pipe<std::vector<unsigned long>> qlabels(4);
    auto data_loader = [&qimages, &qlabels, &objs, &numPeople, &samplesPerId](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;
        while(qimages.is_enabled())
        {
            try
            {
                LoadMiniBatch(numPeople, samplesPerId, rnd, objs, images, labels);
                qimages.enqueue(images);
                qlabels.enqueue(labels);
            }
            catch(std::exception& e)
            {
                std::cout << "EXCEPTION IN LOADING DATA" << std::endl;
                std::cout << e.what() << std::endl;
            }
        }
    };
    // Run the data_loader from as many threads as possible. The number of threads
    // is relative to the number of CPU cores we have.
    uint32_t cores = System::GetCores();
    std::vector<std::thread> data_loader_threads;
    for (int i = 0; i < cores; ++i) {
        std::cout << "Creating data loader thread " << i+1 << "/" << cores << "\r";
        data_loader_threads.emplace_back([data_loader, i](){ data_loader(i+1); });
    }
    std::cout << "Created " << cores << " data loader threads  " << std::endl;

    // Here we do the training. We keep passing mini-batches to the trainer until the
    // learning rate has dropped low enough.
    while(trainer.get_learning_rate() >= 1e-4)
    {
        qimages.dequeue(images);
        qlabels.dequeue(labels);
        trainer.train_one_step(images, labels);
    }

    // Wait for training threads to stop
    trainer.get_net();
    std::cout << "Finished training" << std::endl;

    // stop all the data loading threads and wait for them to terminate.
    qimages.disable();
    qlabels.disable();
    for (int j = 0; j < cores; ++j) {
        std::cout << "Waiting for data loader thread " << j+1 << "/" << cores << "\r";
        data_loader_threads[j].join();
    }
    std::cout << "Stoped " << cores << " data loader threads      " << std::endl;

    return net;
}

void NanoID::Trainer::SaveNet(net_type &net, const string &filename) {
    // Save the network to disk
    net.clean();
    dlib::serialize(filename) << net;
}
