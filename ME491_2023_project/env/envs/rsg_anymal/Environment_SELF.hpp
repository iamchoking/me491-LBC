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
      robot_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf");
      robot_->setName(PLAYER_NAME);
      controller_.setPlayerNum(0);
      controller_.setName(PLAYER_NAME);
      robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

      /// add opponent (with same controller)
      opponent_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
      opponent_->setName("OPPONENT");
      opponentController_.setPlayerNum(1);
      opponentController_.setName("OPPONENT");
      opponent_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

      controller_.setOpponentName("OPPONENT");
      opponentController_.setOpponentName(PLAYER_NAME);

      auto* ground = world_.addGround();
      ground->setName("ground");

      controller_.create(&world_);
      opponentController_.create(&world_);

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

      // std::cout << "Env L87" << std::endl;
      controller_.reset(&world_, theta);
      // std::cout << "Env L90" << std::endl;

      opponentController_.reset(&world_,theta);
      // std::cout << "Env L93" << std::endl;

      timer_ = 0;
    }

    float step(const Eigen::Ref<EigenVec> &action,const Eigen::Ref<EigenVec> &opponentAction) {

      auto curTime = std::chrono::high_resolution_clock::now();

      timer_ += 1;

      controller_.advance(&world_, action);
      opponentController_.advance(&world_, opponentAction);
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] advance controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
        if (server_) server_->lockVisualizationServerMutex();
        world_.integrate();
        if (server_) server_->unlockVisualizationServerMutex();
      }
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] integrate and publish : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      controller_.updateObservation(&world_);
      opponentController_.updateObservation(&world_); //BOOKMARK

      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] update observation for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      controller_.recordReward(&rewards_);    // only needed for "training" controller
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] record reward for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      if(doPrint_){controller_.printStatus(&world_);} // when visualizing, also print some stuff
      return rewards_.sum();
    }

    void observe(Eigen::Ref<EigenVec> ob,Eigen::Ref<EigenVec> opponentOb) {
      auto curTime = std::chrono::high_resolution_clock::now();

      controller_.updateObservation(&world_); // (special function for cube) (NOT INLCUDED IN RAISIMGYMTORCH)
      opponentController_.updateObservation(&world_);
      if(visualizable_ && TIME_VERBOSE){std::cout << "[OBS] observing for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();


      ob = controller_.getObservation().cast<float>();
      opponentOb = opponentController_.getObservation().cast<float>();
      if(visualizable_ && TIME_VERBOSE){std::cout << "[OBS] casting observation : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

    }
    // function to see if player died (modified from [for_test])
    bool player_die(const std::string &name) {
      auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(name));
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

    bool isTerminalState(float &termialReward) {  // this terminalReward is passed to PPO
      auto died = player_die(PLAYER_NAME);
      auto opponentDied = player_die("OPPONENT");

      if (died && opponentDied) {
        winStreak_ = 0;
        termialReward = terminalRewardDraw_;
        return true;
      }

      if (timer_ > 10 * 100) {
        winStreak_ = 0;
        termialReward = terminalRewardDraw_;
        return true;
      }

      if (!died && opponentDied) {
        winStreak_ += 1;
        termialReward = terminalRewardWin_;
        return true;
      }

      if (died && !opponentDied) {
        winStreak_ = 0;
        termialReward = terminalRewardLose_;
        return true;
      }
      return false;
    }

    void curriculumUpdate() {
      if(winStreak_ >= currWinStreak_){
        // cubeMass_ += currMassIncr_;
        // cube_->setMass(cubeMass_);
        // do curricular shit here
        winStreak_ = 0;
        // if(visualizable_){std::cout << "Visualized ENV cube upgraded to: " << cubeMass_ << "KG" << std::endl;}
      }
    };

    void close() { if (server_) server_->killServer(); };

    void setSeed(int seed) {};

    void setSimulationTimeStep(double dt) {
      simulation_dt_ = dt;
      world_.setTimeStep(dt);
    }

    void setControlTimeStep(double dt) { control_dt_ = dt; }

    int getObDim() { return controller_.getObDim(); } // left for back-compat

    int getActionDim() { return controller_.getActionDim(); }

    Eigen::Array2i getObDims(){
      Eigen::Array2i dims;
      dims << controller_.getObDim(),opponentController_.getObDim();
      return dims;
    }

    Eigen::Array2i getActionDims(){
      Eigen::Array2i dims;
      dims << controller_.getActionDim(),opponentController_.getActionDim();
      return dims;
    }

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

    int currWinStreak_;
    double currMassIncr_;

    float terminalRewardWin_ ;
    float terminalRewardLose_;
    float terminalRewardDraw_;

    TRAINING_CONTROLLER controller_;
    TRAINING_CONTROLLER opponentController_;

    raisim::World world_;
    raisim::ArticulatedSystem* robot_;
    raisim::ArticulatedSystem* opponent_;

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