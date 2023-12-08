// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "../../Reward.hpp"

/// include appropriate controller file (AnymalController_????????.hpp)
#include TRAINING_HEADER_FILE_TO_INCLUDE
#include <chrono>

bool TIME_VERBOSE = false;

namespace raisim {

  class ENVIRONMENT {

  public:

    explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
      /// add player
      robot_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf"); //YOU  are always red!
      robot_->setName(PLAYER_NAME);
      controller_.setName(PLAYER_NAME);
      robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

      READ_YAML(int   , trainingMode_         , cfg["training_mode"]);
      READ_YAML(bool  , trainingShuffleInit_  , cfg["training_init_shuffle"]);
      READ_YAML(bool  , trainingDummyOpponent_, cfg["training_dummy_opponent"]);

      /// add opponent (with same controller)
      cubeMass_ = 1.0;
      cube_ = world_.addBox(uniDist_(gen_)*0.5+0.5,uniDist_(gen_)*0.5+0.5,uniDist_(gen_)*1.3+0.2,cubeMass_);
      cube_->setName("OPPONENT");
      controller_.setOpponentName("OPPONENT");

      auto* ground = world_.addGround();
      ground->setName("ground");

      controller_.create(&world_);
      READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
      READ_YAML(double, control_dt_, cfg["control_dt"])

      /// Reward coefficients
      rewards_.initializeFromConfigurationFile (cfg["reward"]);
      READ_YAML(float, terminalRewardWin_ , cfg["reward_win" ]);
      READ_YAML(float, terminalRewardLose_, cfg["reward_lose"]);
      READ_YAML(float, terminalRewardDraw_, cfg["reward_draw"]);

      /// Curriculum Parameters
      READ_YAML(int   , currWinStreak_, cfg["curriculum_win_streak"]);
      READ_YAML(float , currMassIncr_,  cfg["curriculum_mass_incr"]);
      READ_YAML(double, cubeMass_    ,  cfg["curriculum_mass_start"]);
      READ_YAML(bool  , cubeShuffle_ ,  cfg["curriculum_cube_shuffle"]);
      cube_->setMass(cubeMass_);

      /// Stability Training Params
      float stabCubeTime;
      READ_YAML(float, stabCubeTime    ,  cfg["stability_cube_teleport"]);
      READ_YAML(float, stabWinDecay_   ,  cfg["stability_win_decay"]);
      READ_YAML(bool , stabMode_       ,  cfg["stability_mode"]);
      if(stabCubeTime <= 0.0){stabCubeSteps_ = 1000000;}
      else{stabCubeSteps_ = int(stabCubeTime/control_dt_);}

      /// visualize if it is the first environment
      if (visualizable_) {
        server_ = std::make_unique<raisim::RaisimServer>(&world_);
        server_->launchServer();
        server_->focusOn(ground);
        // TODO: play around with server visualizations
        // std::vector<std::string> chartNames;
        // chartNames.emplace_back("OPPONENT");
        // auto barChart = server_->addBarChart("hehehe",chartNames);

        auto cage = server_->addVisualCylinder("cage", 3.0, 0.05);
        cage->setPosition(0,0,0);
      }
    }

    void init() {}

    void reset() {
      auto theta = uniDist_(gen_) * 2 * M_PI;

      if(stabMode_ && winStreak_ > 0){}
      else{controller_.reset(&world_, theta,trainingShuffleInit_);}
      reset_cube();

      timer_ = 0;
    }

    void reset_cube(){
      // world_.removeObject(reinterpret_cast<Object*>(cube_));
      int prevIndex =cube_->getIndexInWorld();
      world_.removeObject(cube_);
      cube_ = world_.addBox(uniDist_(gen_)*0.5+0.5,uniDist_(gen_)*0.5+0.5,uniDist_(gen_)*1.3+0.2,cubeMass_);
      // cube dims: x: 0.5~1.0, y: 0.5~1.0, z: 0.2~1.5
      cube_->setName("OPPONENT");
      controller_.resetOpponentIndex("OPPONENT");
      // std::cout << "cube index updated from " << prevIndex << " to " << cube_->getIndexInWorld() << std::endl;

      /// put back cube (random position)
      Eigen::Vector3d cubePos;
      Vec<3> robotPos;
      robot_->getPosition(0,robotPos);
      cubePos << uniDist_(gen_)*5-2.5, uniDist_(gen_)*5-2.5, cube_->getDim()(2)/2;
      while(cubePos.head(2).norm() > 2.5 || (cubePos.head(2)-robotPos.e().head(2)).norm() < 1 ){
        cubePos << uniDist_(gen_)*5-2.5, uniDist_(gen_)*5-2.5, cube_->getDim()(2)/2;
      }
      cube_->setPosition(cubePos);

      /// put back cube (fixed position)
      // cube_->setPosition(0,0.5,1);

      /// put back cube (random angle)
      auto cubeAngle = uniDist_(gen_)*2*M_PI;
      cube_->setOrientation(cos((cubeAngle - M_PI) / 2), 0, 0, sin((cubeAngle - M_PI) / 2));

      /// put back cube (fixed angle)
      // cube_->setOrientation(1, 0, 0, 0);

      /// zero out velocity
      Vec<3> zeroVel;
      zeroVel.setZero();
      cube_->setVelocity(zeroVel,zeroVel);

    }

    float step(const Eigen::Ref<EigenVec> &action) {

      auto curTime = std::chrono::high_resolution_clock::now();

      timer_ += 1;
      controller_.advance(&world_, action);
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] advance controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
        if (server_) server_->lockVisualizationServerMutex();
        world_.integrate();
        if (server_) server_->unlockVisualizationServerMutex();
      }
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] integrate and publish : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      // controller_.updateObservationCube(&world_); // S' (special function for cube)
      controller_.updateObservation(&world_);
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] update observation for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      controller_.recordReward(&rewards_);    // R
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] record reward for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      if(stabMode_ && timer_ % stabCubeSteps_ == 0){reset_cube();}

      if(doPrint_){controller_.printStatus(&world_);} // when visualizing, also print some stuff
      return rewards_.sum();
    }

    void observe(Eigen::Ref<EigenVec> ob) {
      auto curTime = std::chrono::high_resolution_clock::now();

      // controller_.updateObservationCube(&world_); // (special function for cube) (NOT INLCUDED IN RAISIMGYMTORCH)
      controller_.updateObservation(&world_);
      if(visualizable_ && TIME_VERBOSE){std::cout << "[OBS] observing for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();


      ob = controller_.getObservation().cast<float>();
      if(visualizable_ && TIME_VERBOSE){std::cout << "[OBS] casting observation : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

    }
    // function to see if player died (modified from [for_test])
    bool player_die() {
      auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER_NAME));
      /// base contact with ground
      for(auto& contact: anymal->getContacts()) {
        if(contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
           contact.getlocalBodyIndex() == anymal->getBodyIdx("base")) {
          return true; //player dies by "base" touching the ground
        }
      }
      /// get out of the cage
      int gcDim = int(anymal->getGeneralizedCoordinateDim());
      Eigen::VectorXd gc;
      gc.setZero(gcDim);
      gc = anymal->getGeneralizedCoordinate().e();
      if (gc.head(2).norm() > 3) { // norm of x,y larger than 3
        return true;
      }
      return false;
    }

    /// function to see if the cube died
    bool cube_die() {
      /// get out of the cage
      Eigen::Vector3d gc;
      gc = cube_->getPosition();
      if (gc.head(2).norm() > 3) { // coordinate larger than 3
        return true;
      }
      return false;
    }

    bool isTerminalState(float &termialReward) {  // this terminalReward is passed to PPO

      if (player_die() && cube_die()) {
        winStreak_ = 0;
        termialReward = terminalRewardDraw_;
        return true;
      }

      if (timer_ > 10 * 100) {
        if(stabMode_){winStreak_ = 1;} // to prevent resetting robot on timeout
        else{winStreak_ = 0;}
        termialReward = terminalRewardDraw_;
        return true;
      }

      if (!player_die() && cube_die()) {
        winStreak_ += 1;
        if(stabMode_){ termialReward = terminalRewardWin_*stabWinDecay_;} // less reward for winning (one win and death results in negative net reward)
        else{termialReward = terminalRewardWin_;}
        return true;
      }

      if (player_die() && !cube_die()) {
        winStreak_ = 0;
        termialReward = terminalRewardLose_;
        return true;
      }
      return false;
    }

    void curriculumUpdate() {
      if(winStreak_ >= currWinStreak_){
        cubeMass_ += currMassIncr_;
        cube_->setMass(cubeMass_);
        winStreak_ = 0;
        if(visualizable_){std::cout << "Visualized ENV cube upgraded to: " << cubeMass_ << "KG" << std::endl;}
      }
    };

    void close() { if (server_) server_->killServer(); };

    void setSeed(int seed) {};

    void setSimulationTimeStep(double dt) {
      simulation_dt_ = dt;
      world_.setTimeStep(dt);
    }
    void setControlTimeStep(double dt) { control_dt_ = dt; }

    int getObDim() { return controller_.getObDim(); }

    int getActionDim() { return controller_.getActionDim(); }

    double getControlTimeStep() { return control_dt_; }

    double getSimulationTimeStep() { return simulation_dt_; }

    raisim::World *getWorld() { return &world_; }

    void turnOffVisualization() {
      server_->hibernate();
      doPrint_=false;
    }

    void turnOnVisualization() {
      server_->wakeup();
      doPrint_=true;
    }

    void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

    void stopRecordingVideo() { server_->stopRecordingVideo(); }

    raisim::Reward& getRewards() { return rewards_; }

  private:
    int timer_ = 0;
    int winStreak_ = 0;

    bool visualizable_ = false;
    bool doPrint_ = false;

    double cubeMass_ = 0.5;

    int trainingMode_;
    bool trainingDummyOpponent_;
    bool trainingShuffleInit_;

    int currWinStreak_;
    double currMassIncr_;
    bool cubeShuffle_;

    bool stabMode_;
    int stabCubeSteps_;
    float stabWinDecay_;

    float terminalRewardWin_ ;
    float terminalRewardLose_;
    float terminalRewardDraw_;
    TRAINING_CONTROLLER controller_;
    raisim::Box* cube_;

    raisim::World world_;
    raisim::ArticulatedSystem* robot_;

    raisim::Reward rewards_;
    double simulation_dt_ = 0.001;
    double control_dt_ = 0.01;
    std::unique_ptr<raisim::RaisimServer> server_;
    thread_local static std::uniform_real_distribution<double> uniDist_;
    thread_local static std::mt19937 gen_;
  };
  thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
  thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}