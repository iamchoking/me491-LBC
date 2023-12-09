// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

// CHRIS: modes added!

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

      READ_YAML(int   , trainingMode_         , cfg["training_mode"]);
      READ_YAML(bool  , trainingShuffleInit_  , cfg["training_init_shuffle"]);
      READ_YAML(bool  , trainingDummyOpponent_, cfg["training_dummy_opponent"]);

      if(trainingMode_ == 0) {
        /// add opponent (with same controller)
        cubeMass_ = 1.0;
        cube_ = world_.addBox(uniDist_(gen_)*0.5+0.5,uniDist_(gen_)*0.5+0.5,uniDist_(gen_)*1.3+0.2,cubeMass_);
        cube_->setName("OPPONENT");
      }
      else{
        opponent_ = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_blue.urdf");
        opponent_->setName("OPPONENT");
        opponentController_.setPlayerNum(1);
        opponentController_.setName("OPPONENT");
        opponent_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

        opponentController_.setOpponentName(PLAYER_NAME);
      }

      controller_.setOpponentName("OPPONENT");
      auto* ground = world_.addGround();
      ground->setName("ground");

      controller_.create(&world_);
      if(trainingMode_ == 1) {
        opponentController_.create(&world_);
      }

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
      if(trainingMode_ == 0) {
        cubeMass_ = std::max(0.5,uniDist_(gen_)*5-2.5+cubeMass_); //give a 2.5kg spread for diverse samples
        cube_->setMass(cubeMass_);
      }

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
        // std::vector<std::string> chartNames;
        // chartNames.emplace_back("OPPONENT");
        // auto barChart = server_->addBarChart("hehehe",chartNames);

        auto cage = server_->addVisualCylinder("cage", 3.0, 0.05);
        cage->setPosition(0,0,0);
      }

      metrics_["cube_weight"]      = 0.0;
      metrics_["consecutive_wins"] = 0.0;
      metrics_["win_interval"]     = 0.0;
      metrics_["lose_interval"]    = 0.0;
      metrics_["draw_interval"]    = 0.0;
      metrics_["win_falldown_100"]      = 0.0;
      metrics_["win_pushout_100"]       = 0.0;
      metrics_["lose_falldown_100"]     = 0.0;
      metrics_["lose_pushout_100"]      = 0.0;
      metrics_["win_100"]          = 0.5;
      metrics_["draw_100"]         = 0.5;
      metrics_["lose_100"]         = 0.5;
    }

    void init() {}

    void reset() {
      auto theta = uniDist_(gen_) * 2 * M_PI;

      if(trainingMode_ == 0){
        if(stabMode_ && winStreak_ > 0){}
        else{controller_.reset(&world_, theta,trainingShuffleInit_);}
        reset_cube();
      }
      else {
        // std::cout << "Env L87" << std::endl;
        controller_.reset(&world_, theta,trainingShuffleInit_);
        // std::cout << "Env L90" << std::endl;

        opponentController_.reset(&world_, theta,trainingShuffleInit_);
        // std::cout << "Env L93" << std::endl;
      }
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

    float step(const Eigen::Ref<EigenVec> &action,const Eigen::Ref<EigenVec> &opponentAction) {
      // std::cout << "Env L170" << std::endl;

      auto curTime = std::chrono::high_resolution_clock::now();

      timer_ += 1;
      // std::cout << "Env L176" << std::endl;

      controller_.advance(&world_, action);
      if(trainingMode_ == 1){opponentController_.advance(&world_, opponentAction);}

      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] advance controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      // std::cout << "Env L184" << std::endl;

      for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
        if (server_) server_->lockVisualizationServerMutex();
        world_.integrate();
        if (server_) server_->unlockVisualizationServerMutex();
      }
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] integrate and publish : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      controller_.updateObservation(&world_);
      if(trainingMode_ == 1){opponentController_.updateObservation(&world_);} //BOOKMARK
      // std::cout << "Env L87" << std::endl;

      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] update observation for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      // std::cout << "Env L87" << std::endl;

      controller_.recordReward(&rewards_);    // only needed for "training" controller
      if(visualizable_ && TIME_VERBOSE){std::cout << "[STEP] record reward for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();

      if((trainingMode_ == 0) && stabMode_ && timer_ % stabCubeSteps_ == 0){reset_cube();}

      if(doPrint_){controller_.printStatus(&world_);} // when visualizing, also print some stuff
      return rewards_.sum();
    }

    void observe(Eigen::Ref<EigenVec> ob,Eigen::Ref<EigenVec> opponentOb) {
      auto curTime = std::chrono::high_resolution_clock::now();

      controller_.updateObservation(&world_); // (special function for cube) (NOT INLCUDED IN RAISIMGYMTORCH)
      if(trainingMode_ == 1){opponentController_.updateObservation(&world_);}

      if(visualizable_ && TIME_VERBOSE){std::cout << "[OBS] observing for controller : " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - curTime).count() << std::endl;}
      curTime = std::chrono::high_resolution_clock::now();


      ob = controller_.getObservation().cast<float>();
      if(trainingMode_ == 1){opponentOb = opponentController_.getObservation().cast<float>();}

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
          terminalFactor_ = 1;
          return true; //player dies by "base" touching the ground
        }
      }
      /// get out of the cage
      int gcDim = int(anymal->getGeneralizedCoordinateDim());
      Eigen::VectorXd gc;
      gc.setZero(gcDim);
      gc = anymal->getGeneralizedCoordinate().e();
      if (gc.head(2).norm() > 3) { // norm of x,y larger than 3
        terminalFactor_ = 2;
        return true;
      }
      return false;
    }

    /// function to see if the cube died
    bool cube_die() {
      /// get out of the cage
      if(cube_->getPosition().head(2).norm() > 3){
        terminalFactor_ = 2;
        return true;
      }
      return false;
    }

    void processWin(){
      metrics_["win_interval"] = world_.getWorldTime() - timeLastWin_;
      timeLastWin_ = world_.getWorldTime();
      metrics_["win_100" ] = metrics_["win_100" ]*0.99+0.01;
      metrics_["lose_100"] = metrics_["lose_100"]*0.99+0.00;
      metrics_["draw_100"] = metrics_["draw_100"]*0.99+0.00;
      metrics_["consecutive_wins"] += 1.0;

      if(terminalFactor_ == 1){
        metrics_["win_falldown_100"] = metrics_["win_falldown_100"]*0.99+0.01;
        metrics_["win_pushout_100"]  = metrics_["win_pushout_100"]*0.99+0.00;
      }
      else{
        metrics_["win_falldown_100"] = metrics_["win_falldown_100"]*0.99+0.00;
        metrics_["win_pushout_100"]  = metrics_["win_pushout_100"]*0.99+0.01;
      }
    }

    void processLose(){
      metrics_["lose_interval"] = world_.getWorldTime() - timeLastLose_;
      timeLastLose_ = world_.getWorldTime();
      metrics_["win_100" ] = metrics_["win_100" ]*0.99+0.00;
      metrics_["lose_100"] = metrics_["lose_100"]*0.99+0.01;
      metrics_["draw_100"] = metrics_["draw_100"]*0.99+0.00;
      metrics_["consecutive_wins"] = 0.0;
      if(terminalFactor_ == 1){
        metrics_["lose_falldown_100"] = metrics_["lose_falldown_100"]*0.99+0.01;
        metrics_["lose_pushout_100"]  = metrics_["lose_pushout_100"]*0.99+0.00;
      }
      else{
        metrics_["lose_falldown_100"] = metrics_["lose_falldown_100"]*0.99+0.00;
        metrics_["lose_pushout_100"]  = metrics_["lose_pushout_100"]*0.99+0.01;
      }
    }

    void processDraw(){
      metrics_["draw_interval"] = world_.getWorldTime() - timeLastDraw_;
      timeLastDraw_ = world_.getWorldTime();
      metrics_["win_100" ] = metrics_["win_100" ]*0.99+0.00;
      metrics_["lose_100"] = metrics_["lose_100"]*0.99+0.00;
      metrics_["draw_100"] = metrics_["draw_100"]*0.99+0.01;
      metrics_["consecutive_wins"] = 0.0;
    }

    bool isTerminalState(float &termialReward) {  // this terminalReward is passed to PPO
      auto died = player_die(PLAYER_NAME);
      bool opponentDied;
      if(trainingMode_ == 0){opponentDied = cube_die();}
      else{opponentDied = player_die("OPPONENT");}
      controller_.recordTerminal(&rewards_,0.0f); // need to flush the "terminal" reward for next cycle

      if (died && opponentDied) { // draw (same-time)
        winStreak_ = 0;
        termialReward = terminalRewardDraw_;
        processDraw();
        return true;
      }

      if (timer_ > 10 * 100) { // draw (timeout)
        terminalFactor_ = 0;
        winStreak_ = 0;
        termialReward = terminalRewardDraw_;
        processDraw();
        return true;
      }

      if (!died && opponentDied) { // win
        winStreak_ += 1;
        processWin();

        if(stabMode_){ // stability mode: win does NOT terminate episode! (actions before win needs to backprop to actions after)
          controller_.recordTerminal(&rewards_,terminalRewardWin_*stabWinDecay_);
          reset();
          return false;
        }
        else {
          termialReward = terminalRewardWin_;
          return true;
        }
      }

      if (died && !opponentDied) { // lose
        winStreak_ = 0;
        termialReward = terminalRewardLose_;
        processLose();
        return true;
      }
      return false;
    }

    void curriculumUpdate() {
      if(trainingMode_ == 0) {
        if (winStreak_ >= currWinStreak_) {
          cubeMass_ += currMassIncr_;
          cube_->setMass(cubeMass_);
          winStreak_ = 0;
          if (visualizable_) { std::cout << "Visualized ENV cube upgraded to: " << cubeMass_ << "KG" << std::endl; }
        }
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
      if(trainingMode_ == 0){dims << controller_.getObDim(),1;}
      else {dims << controller_.getObDim(), opponentController_.getObDim();}
      return dims;
    }

    Eigen::Array2i getActionDims(){
      Eigen::Array2i dims;
      if(trainingMode_ == 0){dims << controller_.getActionDim(),1;}
      else {dims << controller_.getActionDim(), opponentController_.getActionDim();}
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

    std::map<std::string,double> getMetrics(){
      // metrics to calculate
      if(trainingMode_==0){metrics_["cube_weight"]=cubeMass_;}
      return metrics_;
    }

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
    TRAINING_CONTROLLER opponentController_;

    raisim::World world_;
    raisim::ArticulatedSystem* robot_;
    raisim::ArticulatedSystem* opponent_;
    raisim::Box* cube_;

    raisim::Reward rewards_;

    double simulation_dt_ = 0.001;
    double control_dt_ = 0.01;

    std::map<std::string,double> metrics_;
    unsigned long timerAbsolute;
    double timeLastWin_  = 0;
    double timeLastLose_ = 0;
    double timeLastDraw_ = 0;
    int terminalFactor_  = 0; // reason the game ended (0: timeout, 1: falldown, 2: pushout)

    std::unique_ptr<raisim::RaisimServer> server_;
    thread_local static std::uniform_real_distribution<double> uniDist_;
    thread_local static std::mt19937 gen_;
  };
  thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
  thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}