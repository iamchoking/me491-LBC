//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "omp.h"
#include "Yaml.hpp"

namespace raisim {

  int THREAD_COUNT;

  template<class ChildEnvironment>
  class VectorizedEnvironment {

  public:

    explicit VectorizedEnvironment(std::string resourceDir, std::string cfg, bool normalizeObservation=true)
      : resourceDir_(resourceDir), cfgString_(cfg), normalizeObservation_(normalizeObservation) {

      // std::cout << "VectorizedEnv L23" << std::endl;

      Yaml::Parse(cfg_, cfg);

      if(&cfg_["render"])
        render_ = cfg_["render"].template As<bool>();
      // std::cout << "VectorizedEnv L30" << std::endl;

      init();
    }

    ~VectorizedEnvironment() {
      for (auto *ptr: environments_)
        delete ptr;
    }

    const std::string& getResourceDir() const { return resourceDir_; }
    const std::string& getCfgString() const { return cfgString_; }

    void init() {

      // std::cout << "VectorizedEnv L45" << std::endl;

      THREAD_COUNT = cfg_["num_threads"].template As<int>();
      omp_set_num_threads(THREAD_COUNT);
      num_envs_ = cfg_["num_envs"].template As<int>();

      // std::cout << "VectorizedEnv L51" << std::endl;

      environments_.reserve(num_envs_);
      rewardInformation_.reserve(num_envs_);
      for (int i = 0; i < num_envs_; i++) {
        environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
        environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
        environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
        rewardInformation_.push_back(environments_.back()->getRewards().getStdMap());
      }

      // std::cout << "VectorizedEnv L62" << std::endl;


      for (int i = 0; i < num_envs_; i++) {
        // only the first environment is visualized
        environments_[i]->init();
        environments_[i]->reset();
      }

      // std::cout << "VectorizedEnv L71" << std::endl;


      obDim_ = environments_[0]->getObDims()(0); //(presumably) same input dims for player and opponent
      opponentObDim_ = environments_[0]->getObDims()(1); //(presumably) same input dims for player and opponent

      actionDim_ = environments_[0]->getActionDims()(0);
      opponentActionDim_ = environments_[0]->getActionDims()(1);

      RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")

      // std::cout << "VectorizedEnv L82" << std::endl;

      if (normalizeObservation_) {
        obMean_.setZero(obDim_);
        obVar_.setOnes(obDim_);
        recentMean_.setZero(obDim_);
        recentVar_.setZero(obDim_);
        delta_.setZero(obDim_);
        epsilon_.setZero(obDim_);
        epsilon_.setConstant(1e-8);

        opponentObMean_.setZero(opponentObDim_);
        opponentObVar_.setOnes(opponentObDim_);
        opponentRecentMean_.setZero(opponentObDim_);
        opponentRecentVar_.setZero(opponentObDim_);
        opponentDelta_.setZero(opponentObDim_);
        opponentEpsilon_.setZero(opponentObDim_);
        opponentEpsilon_.setConstant(1e-8);
      }

      // std::cout << "VectorizedEnv L102" << std::endl;

    }

    // resets all environments and returns observation
    void reset() {
      for (auto env: environments_)
        env->reset();
    }

    void observe(Eigen::Ref<EigenRowMajorMat> &ob, Eigen::Ref<EigenRowMajorMat> &opponentOb, bool updateStatistics) {
#pragma omp parallel for schedule(auto)
      for (int i = 0; i < num_envs_; i++)
        environments_[i]->observe(ob.row(i),opponentOb.row(i));

      if (normalizeObservation_)
        updateObservationStatisticsAndNormalize(ob,opponentOb, updateStatistics);
    }


    void step(Eigen::Ref<EigenRowMajorMat> &action,
              Eigen::Ref<EigenRowMajorMat> &opponentAction,
              Eigen::Ref<EigenVec> &reward,
              Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
      for (int i = 0; i < num_envs_; i++)
        perAgentStep(i, action, opponentAction, reward, done);
    }

    void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
    void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
    void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
    void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }

    void getObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float &count, Eigen::Ref<EigenVec> &opponentMean, Eigen::Ref<EigenVec> &opponentVar, float &opponentCount) {
      mean = obMean_; var = obVar_; count = obCount_;
      opponentMean = opponentObMean_; opponentVar = opponentObVar_; opponentObCount_ = opponentCount;
    }

    void setObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float &count,Eigen::Ref<EigenVec> &opponentMean, Eigen::Ref<EigenVec> &opponentVar,float opponentCount) {
      obMean_ = mean; obVar_ = var; obCount_ = count;
      opponentObMean_ = mean; opponentObVar_ = opponentVar; opponentObCount_ = opponentCount;
    }

    void setSeed(int seed) {
      int seed_inc = seed;

#pragma omp parallel for schedule(auto)
      for(int i=0; i<num_envs_; i++)
        environments_[i]->setSeed(seed_inc++);
    }

    void close() {
      for (auto *env: environments_)
        env->close();
    }

    void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
      for (int i = 0; i < num_envs_; i++) {
        float terminalReward;
        terminalState[i] = environments_[i]->isTerminalState(terminalReward);
      }
    }

    void setSimulationTimeStep(double dt) {
      for (auto *env: environments_)
        env->setSimulationTimeStep(dt);
    }

    void setControlTimeStep(double dt) {
      for (auto *env: environments_)
        env->setControlTimeStep(dt);
    }

    int getObDim() { return obDim_; }
    int getActionDim() { return actionDim_; }

    Eigen::Array2i getObDims(){
      Eigen::Array2i dims;
      dims << obDim_,opponentObDim_;
      return dims;
    }
    Eigen::Array2i getActionDims(){
      Eigen::Array2i dims;
      dims << actionDim_,opponentActionDim_;
      return dims;
    }

    int getNumOfEnvs() { return num_envs_; }

    ////// optional methods //////
    void curriculumUpdate() {
      for (auto *env: environments_)
        env->curriculumUpdate();
    };

    const std::vector<std::map<std::string, float>>& getRewardInfo() { return rewardInformation_; }

    void getMetrics(std::map<std::string,double> &metric) {
    std::map<std::string,double> tempMetric = environments_[0]->getMetrics();
    for (auto it = tempMetric.begin(); it != tempMetric.end(); ++it) {metric[it->first] = 0.0;}

#pragma omp parallel for schedule(auto)

      for(int i=0; i<num_envs_; i++) {
        tempMetric = environments_[i]->getMetrics();
        for (auto it = tempMetric.begin(); it != tempMetric.end(); ++it) {
          metric[it->first] += it->second * (1.0/num_envs_);
        }
      }
    }

  private:
    void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob,Eigen::Ref<EigenRowMajorMat> opponentOb,bool updateStatistics) {
      if (updateStatistics) {
        ///PLAYER
        recentMean_ = ob.colwise().mean();
        recentVar_ = (ob.rowwise() - recentMean_.transpose()).colwise().squaredNorm() / num_envs_;

        delta_ = obMean_ - recentMean_;
        for(int i=0; i<obDim_; i++)
          delta_[i] = delta_[i]*delta_[i];

        float totCount = obCount_ + num_envs_;

        obMean_ = obMean_ * (obCount_ / totCount) + recentMean_ * (num_envs_ / totCount);
        obVar_ = (obVar_ * obCount_ + recentVar_ * num_envs_ + delta_ * (obCount_ * num_envs_ / totCount)) / (totCount);

        obCount_ = totCount;

        ///OPPONENT
        opponentRecentMean_ = opponentOb.colwise().mean();
        opponentRecentVar_ = (opponentOb.rowwise() - opponentRecentMean_.transpose()).colwise().squaredNorm() / num_envs_;

        opponentDelta_ = opponentObMean_ - opponentRecentMean_;
        for(int i=0; i<opponentObDim_; i++)
          opponentDelta_[i] = opponentDelta_[i]*opponentDelta_[i];

        float opponentTotCount = opponentObCount_ + num_envs_;

        opponentObMean_ = opponentObMean_ * (opponentObCount_ / opponentTotCount) + opponentRecentMean_ * (num_envs_ / opponentTotCount);
        opponentObVar_ = (opponentObVar_ * opponentObCount_ + opponentRecentVar_ * num_envs_ + opponentDelta_ * (opponentObCount_ * num_envs_ / opponentTotCount)) / (opponentTotCount);

        opponentObCount_ = opponentTotCount;

      }

#pragma omp parallel for schedule(auto)
      for(int i=0; i<num_envs_; i++) {
        ///PLAYER
        ob.row(i) = (ob.row(i) - obMean_.transpose()).template cwiseQuotient<>(
          (obVar_ + epsilon_).cwiseSqrt().transpose());

        ///OPPONENT
        opponentOb.row(i) = (opponentOb.row(i) - opponentObMean_.transpose()).template cwiseQuotient<>(
          (opponentObVar_ + opponentEpsilon_).cwiseSqrt().transpose());
      }
    }

    inline void perAgentStep(int agentId,
                             Eigen::Ref<EigenRowMajorMat> &action,
                             Eigen::Ref<EigenRowMajorMat> &opponentAction,
                             Eigen::Ref<EigenVec> &reward,
                             Eigen::Ref<EigenBoolVec> &done) {
      reward[agentId] = environments_[agentId]->step(action.row(agentId),opponentAction.row(agentId));
      rewardInformation_[agentId] = environments_[agentId]->getRewards().getStdMap();

      float terminalReward = 0;
      done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

      if (done[agentId]) {
        environments_[agentId]->reset();
        reward[agentId] += terminalReward;
      }
    }

    std::vector<ChildEnvironment *> environments_;
    std::vector<std::map<std::string, float>> rewardInformation_;

    int num_envs_ = 1;
    int obDim_ = 0, actionDim_ = 0;
    int opponentObDim_ = 0, opponentActionDim_ = 0;

    bool recordVideo_=false, render_=false;
    std::string resourceDir_;
    Yaml::Node cfg_;
    std::string cfgString_;

    /// observation running mean
    bool normalizeObservation_ = true;
    EigenVec obMean_;
    EigenVec obVar_;

    EigenVec opponentObMean_;
    EigenVec opponentObVar_;

    float obCount_ = 1e-4;
    float opponentObCount_ = 1e-4;

    EigenVec recentMean_, recentVar_, delta_;
    EigenVec opponentRecentMean_, opponentRecentVar_, opponentDelta_;

    EigenVec epsilon_;
    EigenVec opponentEpsilon_;
  };


  class NormalDistribution {
  public:
    NormalDistribution() : normDist_(0.f, 1.f) {}

    float sample() { return normDist_(gen_); }
    void seed(int i) { gen_.seed(i); }

  private:
    std::normal_distribution<float> normDist_;
    static thread_local std::mt19937 gen_;
  };
  thread_local std::mt19937 raisim::NormalDistribution::gen_;


  class NormalSampler {
  public:
    NormalSampler(int dim) {
      dim_ = dim;
      normal_.resize(THREAD_COUNT);
      seed(0);
    }

    void seed(int seed) {
      // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
      for (int i = 0; i < THREAD_COUNT; i++)
        normal_[0].seed(i + seed);
    }

    inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                       Eigen::Ref<EigenVec> &std,
                       Eigen::Ref<EigenRowMajorMat> &samples,
                       Eigen::Ref<EigenVec> &log_prob) {
      int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
      for (int agentId = 0; agentId < agentNumber; agentId++) {
        log_prob(agentId) = 0;
        for (int i = 0; i < dim_; i++) {
          const float noise = normal_[omp_get_thread_num()].sample();
          samples(agentId, i) = mean(agentId, i) + noise * std(i);
          log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
        }
        log_prob(agentId) -= float(dim_) * 0.9189385332f;
      }
    }
    int dim_;
    std::vector<NormalDistribution> normal_;
  };

}

#endif //SRC_RAISIMGYMVECENV_HPP
