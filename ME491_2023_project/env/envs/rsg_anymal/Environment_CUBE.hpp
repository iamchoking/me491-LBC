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

#include TRAINING_HEADER_FILE_TO_INCLUDE

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      visualizable_(visualizable) {
    /// add plyer
    auto* robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
    robot->setName(PLAYER_NAME);
    controller_.setName(PLAYER_NAME);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// add CUBE (x,y,z,mass)
    auto* cube = world_.addBox(2,2,2,3);
    cube->setName("CUBE");
    controller_.setOpponentName("CUBE");

    world_.addGround();
    ground->setName("ground");

    controller_.create(&world_);
    READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
    READ_YAML(double, control_dt_, cfg["control_dt"])

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);
    READ_YAML(float, terminalRewardWin_ , cfg["reward_win" ]);
    READ_YAML(float, terminalRewardLose_, cfg["reward_lose"]);
    READ_YAML(float, terminalRewardDraw_, cfg["reward_draw"]);


    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer();
      server_->focusOn(robot);
      auto cage = server_->addVisualCylinder("cage", 3.0, 0.05);
      cage->setPosition(0,0,0);
    }
  }

  void init() {}

  void reset() {
    auto theta = uniDist_(gen_) * 2 * M_PI;
    controller_.reset(&world_, theta);

    /// put back cube (static for now)
    cube->setPosition(1.5,1,1);
  }

  float step(const Eigen::Ref<EigenVec> &action) {
    timer_ += 1;
    controller_.advance(&world_, action);
    for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
      if (server_) server_->lockVisualizationServerMutex();
      world_.integrate();
      if (server_) server_->unlockVisualizationServerMutex();
    }
    controller_.updateObservationCube(&world_); // S' (special function for cube)
    controller_.recordReward(&rewards_);    // R
    return rewards_.sum();
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservationCube(&world_); // (special function for cube)
    ob = controller_.getObservation().cast<float>();
  }
  // function to see if player died (modified from [for_test])
  bool player_die() {
    auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER_NAME));
    /// base contact with ground
    for(auto& contact: anymal->getContacts()) {
      if(contact.getPairObjectIndex() == world_.getObject("ground")->getIndexInWorld() &&
         contact.getlocalBodyIndex() == anymal->getBodyIdx("base")) {
        return true;
      }
    }
    /// get out of the cage
    int gcDim = anymal->getGeneralizedCoordinateDim();
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
    auto cube = reinterpret_cast<raisim::SingleBodyObject *>(world_.getObject("CUBE"));
    /// get out of the cage
    Eigen::Vector3d gc;
    gc = cube->getPosition();
    int gcDim = cube->getPosition();
    gc.setZero(gcDim);
    gc = anymal->getGeneralizedCoordinate().e();
    if (gc.head(2).norm() > 3) { // coordinate larger than 3
      return true;
    }
    return false;
  }

  bool isTerminalState(float &termialReward) {  // this terminalReward is passed to PPO

    if (player_die() && cube_die()) {
      draw += 1;
      terminal += 1;
      termialReward = terminalRewardDraw_
      return true;
    }

    if (timer_ > 10 * 100) {
      draw += 1;
      terminal += 1;
      termialReward = terminalRewardDraw_
      return true;
    }

    if (!player_die() && cube_die()) {
      player_win += 1;
      terminal += 1;
      termialReward = terminalRewardWin_
      return true;
    }

    if (player_die() && !cube_die()) {
      cube_win += 1;
      terminal += 1;
      termialReward = terminalRewardLose_
      return true;
    }
  }

  void curriculumUpdate() {};

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

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  raisim::Reward& getRewards() { return rewards_; }

 private:
  int timer_ = 0;
  int player_win = 0, cube_win = 0, draw = 0, terminal = 0;

  bool visualizable_ = false;
  float terminalRewardWin_ ;
  float terminalRewardLose_;
  float terminalRewardDraw_;
  TRAINING_CONTROLLER controller_;
  raisim::World world_;
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