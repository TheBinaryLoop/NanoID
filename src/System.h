//
// Created by Lukas Eßmann on 02/01/2020.
// Copyright (c) 2020 Lukas Eßmann. All rights reserved.
//

#ifndef NANOID_SYSTEM_H
#define NANOID_SYSTEM_H

#include <thread>
namespace NanoID {
    class System {
    public:
        static unsigned int GetCores();
    };

}

#endif //NANOID_SYSTEM_H
