#include "dem_penetrator_io.h"
#include <cctype>
#include <cstdlib>

DEMPeneCylinderIo::DEMPeneCylinderIo() {
    pene_host = NULL;
    Pos = NULL;
    Vel = NULL;
    AngSpd = NULL;
    Quat = NULL;
}

DEMPeneCylinderIo::~DEMPeneCylinderIo() {
    delete pene_host;
    delete Pos;
    delete Vel;
    delete AngSpd;
    delete Quat;
}

void DEMPeneCylinderIo::LoadInfo() {
    const string Filename = "./Rocks/Penetrator.bt";
    ifstream infile(Filename.c_str());
    if (!infile) {
        cerr << "Error! Cannot open " << Filename << "!" << endl;
        exit(EXIT_FAILURE);
    }

    pene_host = new Cylinder{};
    pene_host->MotionMode = PENETRATOR_CONSTANT_VELOCITY;
    pene_host->WaitForContact = 0;
    pene_host->ContactState = 0;
    Pos = new double3;
    Vel = new double3;
    Quat = new double4;
    AngSpd = new double3;

    // Supported formats:
    //   sphere   radius
    //   cylinder radius halfHeight
    // The original numeric-only cylinder format remains supported.
    string shapeToken;
    infile >> shapeToken;
    string lowerToken = shapeToken;
    transform(lowerToken.begin(), lowerToken.end(), lowerToken.begin(),
              [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (lowerToken == "sphere") {
        pene_host->Shape = PENETRATOR_SPHERE;
        infile >> pene_host->Radius;
        pene_host->Height = pene_host->Radius;
        pene_host->Dirct = make_double3(0.0, 0.0, 1.0);
    } else if (lowerToken == "cylinder") {
        pene_host->Shape = PENETRATOR_CYLINDER;
        infile >> pene_host->Radius >> pene_host->Height;
        infile >> pene_host->Dirct.x >> pene_host->Dirct.y >> pene_host->Dirct.z;
    } else {
        char* parseEnd = NULL;
        pene_host->Radius = strtod(shapeToken.c_str(), &parseEnd);
        if (parseEnd == shapeToken.c_str() || *parseEnd != '\0') {
            cerr << "Error! Unknown penetrator shape '" << shapeToken << "' in " << Filename << "."
                 << endl;
            exit(EXIT_FAILURE);
        }
        pene_host->Shape = PENETRATOR_CYLINDER;
        infile >> pene_host->Height;
        infile >> pene_host->Dirct.x >> pene_host->Dirct.y >> pene_host->Dirct.z;
    }

    infile >> pene_host->Property.x >> pene_host->Property.y >> pene_host->Property.z >>
        pene_host->Property.w;
    infile >> Pos[0].x >> Pos[0].y >> Pos[0].z;
    infile >> Vel[0].x >> Vel[0].y >> Vel[0].z;
    infile >> Quat[0].x >> Quat[0].y >> Quat[0].z >> Quat[0].w;
    infile >> AngSpd[0].x >> AngSpd[0].y >> AngSpd[0].z;

    if (!infile || pene_host->Radius <= 0.0 || pene_host->Property.w <= 0.0) {
        cerr << "Error! Invalid penetrator data in " << Filename << " (radius=" << pene_host->Radius
             << ", mass=" << pene_host->Property.w << ", stream_ok=" << static_cast<bool>(infile)
             << ")." << endl;
        exit(EXIT_FAILURE);
    }

    // Optional final line (legacy files without it remain constant velocity):
    //   motion constant_velocity
    //   motion gravity
    //   motion gravity_after_contact (prescribed approach speed, then dynamic)
    string optionToken;
    if (infile >> optionToken) {
        transform(optionToken.begin(), optionToken.end(), optionToken.begin(),
                  [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (optionToken != "motion") {
            cerr << "Error! Unknown penetrator option '" << optionToken << "' in " << Filename
                 << "." << endl;
            exit(EXIT_FAILURE);
        }

        string motionToken;
        infile >> motionToken;
        transform(motionToken.begin(), motionToken.end(), motionToken.begin(),
                  [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (motionToken == "constant_velocity" || motionToken == "constant" ||
            motionToken == "kinematic") {
            pene_host->MotionMode = PENETRATOR_CONSTANT_VELOCITY;
        } else if (motionToken == "gravity" || motionToken == "dynamic") {
            pene_host->MotionMode = PENETRATOR_GRAVITY;
        } else if (motionToken == "gravity_after_contact") {
            pene_host->MotionMode = PENETRATOR_GRAVITY;
            pene_host->WaitForContact = 1;
        } else {
            cerr << "Error! Unknown penetrator motion mode '" << motionToken << "' in " << Filename
                 << "." << endl;
            exit(EXIT_FAILURE);
        }
    }
}
