//
// Created by Lukas Eßmann on 02/01/2020.
// Copyright (c) 2020 Lukas Eßmann. All rights reserved.
//

#include "System.h"
namespace NanoID {

    unsigned int System::GetCores() {
        return std::thread::hardware_concurrency();
    }

};
