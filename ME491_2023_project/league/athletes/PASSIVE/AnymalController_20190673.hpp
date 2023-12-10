// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"
#include <random>

namespace raisim {

/// change the class name and file name ex) AnymalController_00000000 -> AnymalController_STUDENT_ID
class AnymalController_20190673 {

 public:
  inline bool create(raisim::World *world) {
    world_ = world;
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));
    opponentObjectIdx_ = world->getObject(opponentName_)->getIndexInWorld();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 54;
    // (I said 54 because I did the math wrong on relative pos dimensions XD)

    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionScale_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    obFloat_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionScale_.setConstant(0.1);
    // actionScale_.setConstant(0.2);

    /// indices of links that should not make contact with ground
    footIndices_.push_back(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.push_back(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.push_back(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.push_back(anymal_->getBodyIdx("RH_SHANK"));

    return true;
  }

  inline bool init(raisim::World *world) {
    return true;
  }

  inline bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionScale_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal_->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  inline bool reset(raisim::World *world, double theta,bool randomPos=false, bool randomHeading=false) {
    // theta = 30.0; //uncomment this line for determinism
    if (playerNum_ == 0) {
      // std::cout << "Controller L93" << std::endl;
      if(randomPos){gc_init_.head(3) << (double(random()%10000)/10000.0)*5-2.5,(double(random()%10000)/10000.0)*5-2.5,0.5;}
      else{gc_init_.head(3) << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;}

      if(randomHeading){
        double initialAng = (double(random()%10000)/10000.0)*M_PI*2;
        gc_init_.segment(3,4) << cos(initialAng / 2), 0, 0, sin(initialAng / 2);
      }
      else{         // same as skeleton
        gc_init_.segment(3, 4) << cos((theta - M_PI) / 2), 0, 0, sin((theta - M_PI) / 2);
      }
      // std::cout << "Controller L97" << std::endl;
    }
    else {
      // std::cout << "Controller L103" << std::endl;
      Eigen::Vector3d opponentPos;
      opponentPos << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5; // assume opponent was also spawned with fixed heading

      if(randomPos){
        Eigen::Vector3d initialPos;
        initialPos << (double(random()%10000)/10000.0)*5-2.5,(double(random()%10000)/10000.0)*5-2.5,0.5;
        while ((initialPos - opponentPos).norm() < 1.5) {
          initialPos << (double(random() % 10000) / 10000.0) * 5 - 2.5, (double(random() % 10000) / 10000.0) * 5 - 2.5, 0.5;
        }
        gc_init_.head(3) << initialPos;
      }
      else{
        gc_init_.head(3) << 1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5;
      }
      if(randomHeading){
        double initialAng = (double(random()%10000)/10000.0)*M_PI*2;
        gc_init_.segment(3,4) << cos(initialAng / 2), 0, 0, sin(initialAng / 2);
      }
      else {
        // "face" the opponent
        theta = atan2(opponentPos(1) - gc_init_(1),opponentPos(0) - gc_init_(0));
        gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
      }

      // gc_init_.head(3) << 2.5 * std::cos(theta + M_PI), 2.5 * std::sin(theta + M_PI), 0.5;
      // gc_init_.segment(3,4) << 1,0,0,0; // no "heading correction" (effective heading randomness)
    }

    anymal_->setState(gc_init_, gv_init_);
    return true;
  }

  

  inline void updateObservation(raisim::World *world) {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rotMat_);

    /// var    dims    description

    /// position data  [6]
    ///  r      1      distance from center, in xy plane
    ///  z      1      body height, world coordinates
    /// th      1      angle between xy-plane projection of body x axis and vector from origin to body center
    /// ez_b    3      gravity direction in body coordinates

    /// velocity data  [6]
    ///  rdot   1      movement speed from center, in terms of xy plane
    ///  zdot   1      body height speed, world coordinates
    /// vrad    1      radial speed norm(v_xy - rdot)
    ///  w_b    3      angular velocity in body coordinates

    /// joint data     [24]
    /// gc_.tail(12)
    /// gv_.tail(12)

    /// relative data  [7]
    /// relativePos_  3      relative position in body coordinates
    /// relativeVel_  3      relative velocity (moving frame) in body coordinates
    /// headAng_ 1      angle (x-projection) of self and opponent

    /// contact data   [5]
    /// contFt  4      feet contact (binary)
    /// contOpp 1      opponent contact (binary)

    /// total: 48 dims (I said 54 because I did the math wrong on relative pos dimensions XD)

    bodyLinearVel_ = rotMat_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rotMat_.e().transpose() * gv_.segment(3, 3);
    world->getObject(opponentObjectIdx_)->getPosition(0,opponentPosVec_);
    world->getObject(opponentObjectIdx_)->getOrientation(0,opponentRotMat_);
    world->getObject(opponentObjectIdx_)->getVelocity(0,opponentLinearVelVec_);

    // Eigen::Vector3d relativePos_= rotMat_.e().transpose() * (opponentPosVec_.e() - gc_.segment(0,2));
    Vec<3> relPosVec;
    anymal_->getPositionInBodyCoordinate(0,opponentPosVec_,relPosVec);

    // position data
    auto r = gc_.head(2).norm();
    auto z = gc_(2);
    // th is the angle betweeen (rotMat_.e().col(0).head(2)) and (-(gc_.head(2))
    auto ex_w = rotMat_.e().col(0).head(2);
    auto rx_w = -(gc_.head(2)); // xy-proj of position vector from body to origin, in world coord
    auto th = std::atan2(ex_w(1)*rx_w(0)-ex_w(0)*rx_w(1),ex_w.transpose()*rx_w);

    auto ez_b = rotMat_.e().row(2).transpose();

    // velocity data
    auto rdot = gv_.head(2).transpose()*rx_w;
    auto zdot = gv_(2);
    auto vrad = (gv_.head(2)- rdot*(rx_w)).norm(); // the normal compoenent of gv_.head(2) wrt rx_w)
    auto w_b = rotMat_.e().transpose() * gv_.segment(3,3);

    // relative data
    relativePos_ = relPosVec.e();
    relativeVel_ = rotMat_.e().transpose() * (opponentLinearVelVec_.e() - gv_.segment(0,3));

    Vec<3> oppx_b; // x axis of opponent expressed in body coord.
    anymal_->getPositionInBodyCoordinate(0,opponentRotMat_.col(0),oppx_b);
    if(oppx_b.e().head(2).norm() < 0.01){headAng_ = 0;} // prevent erratic values when enemy x axis is almost alligned with my z axis
    else{headAng_ = std::atan2(oppx_b.e()(1),oppx_b.e()(0));}

    // contact data
    Eigen::Vector4d contFt;
    double contOpp = 0.0;
    contFt.setZero();

    feetContact_[0] = -1;
    feetContact_[1] = -1;
    feetContact_[2] = -1;
    feetContact_[3] = -1;
    opponentContact_ = false;

    for (auto &contact: anymal_->getContacts()) {
      if (contact.getPairObjectIndex() == opponentObjectIdx_) { opponentContact_ = true; contOpp = 1.0;}
      for(int i = 0;i < 4;i++){
        if(contact.getlocalBodyIndex() == footIndices_[i]){
          feetContact_[i]=contact.getIndexInObjectContactList();
          contFt(i)=1.0;
        }
      }
    }

    obDouble_ <<
      //position data
      r,z,th,ez_b,
      // velocity data
      rdot,zdot,vrad,w_b,
      // joint data
      gc_.tail(12),gv_.tail(12),
      // relative data
      relativePos_,relativeVel_,headAng_,
      // contact data
      contFt,contOpp
      ;

    // checing via cout
    // std::cout << "player: " << playerNum_ << std::endl;
    // std::cout << "[6]  position data    : " <<  r << " | " << z << " | " << th << " | " << ez_b.transpose() << std::endl;
    // std::cout << "[6]  velocity data    : " <<  rdot << " | " << zdot << " | " << vrad << " | " << w_b.transpose() << std::endl;
    // std::cout << "[12] joint data (pos) : " << gc_.tail(12).transpose() << std::endl;
    // std::cout << "[12] joint data (vel) : " << gv_.tail(12).transpose() << std::endl;
    // std::cout << "[7]  relative data    : " << relativePos_.transpose() << " | " << relativeVel_.transpose() << " ha: "<< headAng_ << std::endl;
    // std::cout << "[5]  contact (t)      : ft: " << contFt.transpose() << " opp: " << contOpp << "( from " << feetContact_[0] << feetContact_[1] << feetContact_[2] << feetContact_[3] << ")" << std::endl;
    // std::cout << "full obs: " << obDouble_.transpose() << std::endl;
  }

  inline void recordTerminal(Reward *rewards,float terminalReward){ //used to record rewards without a terminal transition
    rewards->record("terminal",terminalReward);
  }

  inline void recordReward(Reward *rewards) { ///(!!!) Setting rewards!
    /// assumed updateObservation is ran before this function is run
    /// (--> ) _gc and _gv is always updated prior to calling this function. (,etc.)

    /// PENALIZE
    // [time]
    rewards->record("time",1.0f);

    // [slipping & no feet contact]
    Vec<3> bContact;
    // Vec<3> contactVel1;
    Vec<3> contactVel2;
    float slipMaxVel = 0;
    float flight = 1.0f;

    for(int i;i<4;i++){
      if(feetContact_[i] != -1){
        flight = 0.0f;
        anymal_->getContactPointVel(feetContact_[i],contactVel2);
        slipMaxVel = std::max(std::min(float(contactVel2.e().head(2).norm()), 1.00f), slipMaxVel); //clipped at 1
      }
    }
    rewards->record("slip",slipMaxVel);
    rewards->record("flight",flight);

    // [too much tilt]
    rewards->record("tilt",float(rotMat_.e().col(2).head(2).norm())); //cosine of tilt angle (1 when z is horizontal)

    // [too much spin (x and y direction)]
    Eigen::Vector3d w_b = rotMat_.e().transpose()*gv_.segment(3,3);
    // std::cout << "w_b : " << w_b(0) << " , " << w_b(1) << std::endl;
    // std::cout << "Reward: " << float(std::min(10.0,std::max(abs(w_b(0)),abs(w_b(1)))))/10.0f << std::endl;
    rewards->record("spin",float(std::min(10.0,std::max(abs(w_b(0)),abs(w_b(1)))))/10.0f); //clipped at 10 rad/s

    // [posiition / speed toward the edge (1m)] "when within 1m of the edge, penalize speed toward the edge"
    Eigen::Vector2d centerUnit = gc_.head(2);
    centerUnit.normalize();
    float edgeSpeed = std::max(-1.0,std::min(3.0,gv_.head(2).dot(centerUnit.transpose())))/3.0; //max is 3, min is -1
    float edgeScore = float(std::max(gc_.head(2).norm()-2.0,0.0)*edgeSpeed); //normalized from to -0.333 to 1

    if(opponentPosVec_.e().head(2).norm() > gc_.head(2).norm()){edgeScore *= 0.3;} // take more risks when doing better
    rewards->record("edge",edgeScore);

    /// PROMOTE

    // ["facing the opponent"]
    Vec<3> relativePosVec_;
    double angle;
    angle = std::atan2(relativePos_(1),relativePos_(0)); //angle: angle between xy-projection of relativePos_ and x axis of body frame
    // std::cout <<"\rfacing angle : " << angle/M_PI*180.0 << std::endl;
    rewards->record("face",float(-abs(angle)/(2*M_PI))); //scaled to (-1 ~ 0)

    // [closing relative velocity]
    Eigen::Vector3d eRelPos = relativePos_;
    eRelPos.normalize();
    // reward velocity facing toward opponent,
    float ram_clip = 2.0f; // -ram_clip~+ram_clip scaled to -1~1
    float ramPoint = std::min(ram_clip,std::max(-ram_clip,(float(eRelPos.dot(rotMat_.e().transpose() * gv_.segment(0, 3)))))) / ram_clip ;
    if(opponentContact_){
      auto opponentMass = world_->getObject(opponentObjectIdx_)->getMass(0);
      ramPoint+=ramPoint*float(1+opponentMass/anymal_->getMass(0))+0.3f; // when in contact, give it bonus + scaled for mass (proportional to resulting momentum)
    } // when in contact, give full mark plus extra (always promote contact)

    rewards->record("ram",ramPoint);

    // [closer to center] (only reward when in center 1m radius)
    rewards->record("center", 1.0-std::min(1.0,gc_.head(2).norm()) );

    // [far away from opponent]
    rewards->record("away",std::min(2.0,relativePos_.norm())/2.0);

    // "T-boning opponent"
    // (only give reward when robot is "facing" the opponent)
    auto tboneScore = 1.0f - float((std::abs(headAng_)-M_PI/2)*(std::abs(headAng_)-M_PI/2) / (M_PI*M_PI/4.0));
    double tboneMultiplier = 0;

    Vec<3> opp_bVec;
    Eigen::Vector3d x_b;
    anymal_->getPositionInBodyCoordinate(0,opponentPosVec_,opp_bVec);
    Eigen::Vector3d opp_b = opp_bVec.e();
    opp_b.normalize();
    x_b << 1,0,0;

    // opponent within 10deg cone and close proximity
    if(opp_b.transpose()*x_b > std::cos(M_PI/18.0) && opponentPosVec_.norm() < 1.5){tboneMultiplier += 1;}

    Vec<3> bod_oVec;
    Eigen::Vector3d x_o;
    Eigen::Vector3d bod_o = opponentRotMat_.transpose()*rotMat_*(-opp_b); // unit vector from o to b, in o frame
    x_o << 1,0,0;
    // I am getting "T-boned"
    if(bod_o.transpose()*x_o > std::cos(M_PI/18.0) && opponentPosVec_.norm() < 1.5){tboneMultiplier -= 1;}

    rewards->record("tbone", tboneScore*tboneMultiplier);

  }

  inline void printStatus(raisim::World *world){

  }

  inline void resetOpponentIndex(const std::string name){
    opponentName_ = name;
    opponentObjectIdx_ = world_->getObject(name)->getIndexInWorld();
  }

  inline const Eigen::VectorXd &getObservation() {
    return obDouble_;
  }

  void setName(const std::string &name) {
    name_ = name;
  }

  void setOpponentName(const std::string &name) {
    opponentName_ = name;
  }

  void setPlayerNum(const int &playerNum) {
    playerNum_ = playerNum;
  }

  inline bool isTerminalState(raisim::World *world) {
    /// not needed for sumo (terminal if there is no contact on feet / other contacts exist)
    // for (auto &contact: anymal_->getContacts()) {
    //   if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
    //     return true;
    //   }
    // }
    return false;
  }

  inline int getObDim() {
    return obDim_;
  }

  inline int getActionDim() {
    return actionDim_;
  }


 private:
  raisim::World* world_;

  std::string name_, opponentName_;
  size_t opponentObjectIdx_;

  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;
  raisim::ArticulatedSystem *anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  raisim::Mat<3, 3> rotMat_;

  Eigen::VectorXd actionMean_, actionScale_, obDouble_; // actionScale_ should be inv. proportional to action stddev in training
  Eigen::VectorXf obFloat_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::vector<size_t> footIndices_;

  raisim::Vec<3>   opponentPosVec_;
  raisim::Mat<3,3> opponentRotMat_;
  raisim::Vec<3>   opponentLinearVelVec_;

  Eigen::Vector3d relativePos_;
  Eigen::Vector3d relativeVel_;

  double headAng_;

  bool    opponentContact_;
  int     feetContact_[4]; //elements are -1 or contact.getIndexInObjectContactList()

  Eigen::VectorXd obPadding_;
  int obDim_ = 0, actionDim_ = 0, obPaddingDim_ = 0;
};

}