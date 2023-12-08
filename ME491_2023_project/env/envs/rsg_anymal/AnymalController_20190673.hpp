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
    obDim_ = 64; // always padded to 64 (for compatibility)
    // ACTUAL DIMS: 42
    obPaddingDim_ = 64-47;
    obPadding_.setZero(obPaddingDim_);

    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    obFloat_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);
    // actionStd_.setConstant(0.2);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    return true;
  }

  inline bool init(raisim::World *world) {
    return true;
  }

  inline bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal_->setPdTarget(pTarget_, vTarget_);
    return true;
  }

  inline bool reset(raisim::World *world, double theta,bool random_heading=false) {
    if (playerNum_ == 0) {
      // std::cout << "Controller L93" << std::endl;

      gc_init_.head(3) << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;                 //position
      if(random_heading){gc_init_.segment(3,4) << 1,0,0,0;} // no "heading correction" (effective heading randomness)
      else{gc_init_.segment(3, 4) << cos((theta - M_PI) / 2), 0, 0, sin((theta - M_PI) / 2);} //rotation
      // std::cout << "Controller L97" << std::endl;

      // gc_init_.head(3) << 2.5 * std::cos(theta), 2.5 * std::sin(theta), 0.5;                 //position
    }
    else {
      // std::cout << "Controller L103" << std::endl;

      if(random_heading){
        Eigen::Vector3d initialPos;
        Eigen::Vector3d opponentPos;
        opponentPos << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
        initialPos << (double(random()%10000)/10000.0)*5-2.5,(double(random()%10000)/10000.0)*5-2.5,0.5;
        while ((initialPos - opponentPos).norm() < 1){
          initialPos << (double(random()%10000)/10000.0)*5-2.5,(double(random()%10000)/10000.0)*5-2.5,0.5;
        }
        gc_init_.head(3) << initialPos;
      }
      else {
        gc_init_.head(3) << 1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5;
      }
      // std::cout << "Controller L106" << std::endl;

      if(random_heading){gc_init_.segment(3,4) << 1,0,0,0;} // no "heading correction" (effective heading randomness)
      else{gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);}
      // std::cout << "Controller L109" << std::endl;


      // gc_init_.head(3) << 2.5 * std::cos(theta + M_PI), 2.5 * std::sin(theta + M_PI), 0.5;
      // gc_init_.segment(3,4) << 1,0,0,0; // no "heading correction" (effective heading randomness)
    }

    anymal_->setState(gc_init_, gv_init_);
    return true;
  }

  inline void updateObservation(raisim::World *world) { /// (!!!) getting observation
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot_);
    bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// if you want use opponent cube`s state, use like below code

    world->getObject(opponentObjectIdx_)->getPosition(0,opponentPos_);
    world->getObject(opponentObjectIdx_)->getOrientation(0,opponentRot_);
    world->getObject(opponentObjectIdx_)->getVelocity(0,opponentLinearVel_);
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // Eigen::Vector3d relativePos = rot_.e().transpose() * (opponentPos_.e() - gc_.segment(0,2));
    Vec<3> relativePos;
    anymal_->getPositionInBodyCoordinate(0,opponentPos_,relativePos);

    Eigen::Vector3d relativeVel = rot_.e().transpose() * (opponentLinearVel_.e() - gv_.segment(0,2));

    auto x_b = rot_.e().coeff(0,0);
    auto y_b = rot_.e().coeff(0,1);

    auto x_o = opponentRot_.e().coeff(0,0);
    auto y_o = opponentRot_.e().coeff(0,1);

    auto headingAngle = std::atan2(y_o,x_o) - std::atan2(y_b,x_b);

    // obPadding_ << Eigen::VectorXd::Random(obPaddingDim_)*5.0; // need to randomize to discourage inference
    // std::cout << obPadding_ << "garbled?" <<std::endl ;

    obDouble_ <<                         /// Name                             Dims
       gc_.head(3)                    /// body position                     3
      ,rot_.e().row(2).transpose()     /// body orientation                 3
      ,gc_.tail(12)                   /// joint angles                     12
      ,bodyLinearVel_, bodyAngularVel_   /// body linear&angular velocity     6
      ,gv_.tail(12)                   /// joint velocity                   12
      ,relativePos.e(),relativeVel       /// relative position and velocity   6
      ,headingAngle                      /// heading angle for players        1
      ,gc_.head(2).norm()             /// center distance                  1
      ,gv_.head(3)                    /// body velocity                    3
      ,obPadding_                        /// padding                          (17)
      ;                                  /// (total+padding)                  47+17
  }

  inline void recordReward(Reward *rewards) { ///(!!!) Setting rewards!
    /// (skeleton) reward for forward running
    /// assumed that _gc and _gv is always updated prior to calling this function.
    bool opponentContact = false;

    /// MASS of opponent (this is mainly for graphing purposes)
    auto opponentMass = world_->getObject(opponentObjectIdx_)->getMass(0);
    rewards->record("class",float(opponentMass));

    /// PENALIZE
    // [time]
    rewards->record("time",float(opponentMass)+1.0f); //this cancels out with class to yield a constant negative reward

    // [slipping & no feet contact]
    Vec<3> bContact;
    // Vec<3> contactVel1;
    Vec<3> contactVel2;
    float slipMaxVel = 0;
    float flight = 1;
    for (auto &contact: anymal_->getContacts()) {
      if(contact.getPairObjectIndex() == opponentObjectIdx_){opponentContact = true;}

      if (footIndices_.find(contact.getlocalBodyIndex()) != footIndices_.end()) { // is a foot contact
        flight = 0;
        // anymal_->getPositionInBodyCoordinate(contact.getlocalBodyIndex(),contact.getPosition(),bContact);
        // anymal_->getVelocity(contact.getlocalBodyIndex(),bContact,contactVel1);
        anymal_->getContactPointVel(contact.getIndexInObjectContactList(),contactVel2);
        // std::cout << "Vel1 : " << contactVel1 << std::endl;
        // std::cout << "Vel2 : " << contactVel2 << std::endl;
        if(contactVel2.e().coeff(2) > 0.01) { // don't count as slipping if contact is moving away (z value larger than zero)
          // flight = 0;
          // slipMaxVel = std::max(std::min(float(contactVel1.e().head(2).norm()), 1.00f), slipMaxVel); //clipped at 1
          slipMaxVel = std::max(std::min(float(contactVel2.e().head(2).norm()), 1.00f), slipMaxVel); //clipped at 1
        }
        // std::cout << "[Contact] idx:" << contact.getlocalBodyIndex() <<" pos: " << std::endl;
        // std::cout << contact.getPosition() << std::endl;
        // std::cout << " velocity: " << wVelocity << " score: " << slipMaxVel << std::endl;
      }
    }
    // if(slipMaxVel < 0.1){slipMaxVel = 0;} // no slipping if speed less than 0.1
    // std::cout << "# of feet in contact:" << numTouch;
    // if(slipMaxVel > 0.3){std::cout << " SLIP (vel: " << slipMaxVel << ")";}
    // if(numTouch == 0){std::cout << " FLIGHT";}
    // std::cout << std::endl;
    rewards->record("slip",slipMaxVel);
    rewards->record("flight",flight);

    // [too much tilt]
    rewards->record("tilt",float(std::pow(rot_(0,2),2.0)) + float(std::pow(rot_(1,2),2.0)));
    // ^ rot_(1,2) x component of unit z vector at 0, rot_(2,2) y component of ~

    // [posiition / speed toward the edge (1m)] "when within 1m of the edge, penalize speed toward the edge"
    Eigen::Vector2d centerUnit = gc_.head(2);
    centerUnit.normalize();
    float edgeSpeed = std::max(-1.0,std::min(3.0,gv_.head(2).dot(centerUnit.transpose())))/3.0; //max is 3, min is -1
    float edgeScore = float(std::max(gc_.head(2).norm()-2.0,0.0)*edgeSpeed); //normalized from to -0.333 to 1

    if(opponentPos_.e().head(2).norm() > gc_.head(2).norm()){edgeScore *= 0.3;} // take more risks when doing better
    rewards->record("edge",edgeScore);

    /// PROMOTE

    // ["facing the opponent"]
    Vec<3> relPos;
    double angle;
    anymal_->getPositionInBodyCoordinate(0,opponentPos_,relPos); //relPos: vector of opponent position in world frame
    angle = std::atan2(relPos.e().coeff(1),relPos.e().coeff(0)); //angle: angle between xy-projection of relPos and x axis of body frame
    // std::cout <<"\rfacing angle : " << angle/M_PI*180.0 << std::endl;
    rewards->record("face",float(-abs(angle)/(2*M_PI))); //scaled to (-1 ~ 0)

    // [closing relative velocity]
    Eigen::Vector3d eRelPos = relPos.e();
    eRelPos.normalize();
    // reward velocity facing toward opponent,
    float ramPoint = std::min(3.0f,std::max(-3.0f,(float(eRelPos.dot(rot_.e().transpose() * gv_.segment(0, 3)))))) / 3.0f ; //scaled to -1~1
    if(opponentContact){
      ramPoint+=ramPoint*float(1+opponentMass/anymal_->getMass(0))+0.3f; // when in contact, give it bonus + scaled for mass (proportional to resulting momentum)
    } // when in contact, give full mark plus extra (always promote contact)

    rewards->record("ram",ramPoint);

    // [closer to center]
    rewards->record("center", gc_.head(2).norm());

    // "T-boning opponent" TODO

  }

  inline void printStatus(raisim::World *world){
    // "facing the opponent"
    // Vec<3> relPos;
    // double angle;
    // anymal_->getPositionInBodyCoordinate(0,opponentPos_,relPos); //relPos: vector of opponent position in world frame
    // angle = std::atan2(relPos.e().coeff(1),relPos.e().coeff(0)); //angle: angle between xy-projection of relPos and x axis of body frame
    // // std::cout <<"\rfacing angle : " << angle/M_PI*180.0 << std::endl;
    // std::cout << "angle: " << angle/M_PI*180 << "  ";
    // std::cout << float(-abs(angle)/(M_PI)) << std::endl; //scaled to (-1 ~ 0)
    // std::cout << opponentPos_<<std::endl;
    // std::cout << float(std::pow(rot_(0,2),2.0)) + float(std::pow(rot_(1,2),2.0)) << std::endl;
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
  raisim::Mat<3, 3> rot_;

  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::VectorXf obFloat_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;

  raisim::Vec<3>   opponentPos_;
  raisim::Mat<3,3> opponentRot_;
  raisim::Vec<3>   opponentLinearVel_;

  Eigen::VectorXd obPadding_;
  int obDim_ = 0, actionDim_ = 0, obPaddingDim_ = 0;
};

}