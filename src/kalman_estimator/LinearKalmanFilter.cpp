//
// Created by qiayuan on 2022/7/24.
//

#include <pinocchio/fwd.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/joint/joint-composite.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <urdf_parser/urdf_parser.h>


#include "kalman_estimator/LinearKalmanFilter.h"


namespace legged {

KalmanFilterEstimate::KalmanFilterEstimate(std::string robot_name) {

  ReadKalmanYaml(robot_name);
  std::string urdfFilePath = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/assets/" + robot_name + "_description/" + robot_name + ".urdf";
  std::cout<< "[Loading] urdfFilePath: " << urdfFilePath << std::endl;
  this->pinocchioInterfacePtr =
      std::make_unique<PinocchioInterface>(createPinocchioInterface(urdfFilePath, this->robot_info.joint_names));
  std::cout << "[Loading] pinocchio : " << *this->pinocchioInterfacePtr << std::endl;
  this->eeKinematicsPtr = std::make_shared<PinocchioEndEffectorKinematics>(*this->pinocchioInterfacePtr, this->robot_info.foot_names);
    
  this->eeKinematicsPtr->setPinocchioInterface(*this->pinocchioInterfacePtr);

  this->rbdState.setZero(2 * this->robot_info.generalizedCoordinatesNum);
  // left force(6) + right force(6) + left linear norm(1) + right linear norm(1) + left linear angular norm(1) + right linear angular norm(1)
  this->est_contact_force.resize(12+4);
  this->est_contact_force.fill(50);
  this->pSCg_z_inv_last.resize(this->robot_info.generalizedCoordinatesNum);
  this->pSCg_z_inv_last.setZero();

  this->cmd_torque.resize(this->robot_info.actuatedDofNum);
  this->cmd_torque.setZero();
  
  xHat_.setZero();
  ps_.setZero();
  vs_.setZero();
  a_.setZero();
  a_.block(0, 0, 3, 3) = Eigen::Matrix<scalar_t, 3, 3>::Identity();
  a_.block(3, 3, 3, 3) = Eigen::Matrix<scalar_t, 3, 3>::Identity();
  a_.block(6, 6, 12, 12) = Eigen::Matrix<scalar_t, 12, 12>::Identity();
  b_.setZero();

  Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> c1(3, 6);
  c1 << Eigen::Matrix<scalar_t, 3, 3>::Identity(), Eigen::Matrix<scalar_t, 3, 3>::Zero();
  Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> c2(3, 6);
  c2 << Eigen::Matrix<scalar_t, 3, 3>::Zero(), Eigen::Matrix<scalar_t, 3, 3>::Identity();
  c_.setZero();
  c_.block(0, 0, 3, 6) = c1;
  c_.block(3, 0, 3, 6) = c1;
  c_.block(6, 0, 3, 6) = c1;
  c_.block(9, 0, 3, 6) = c1;
  c_.block(0, 6, 12, 12) = -Eigen::Matrix<scalar_t, 12, 12>::Identity();
  c_.block(12, 0, 3, 6) = c2;
  c_.block(15, 0, 3, 6) = c2;
  c_.block(18, 0, 3, 6) = c2;
  c_.block(21, 0, 3, 6) = c2;
  c_(27, 17) = 1.0;
  c_(26, 14) = 1.0;
  c_(25, 11) = 1.0;
  c_(24, 8) = 1.0;
  p_.setIdentity();
  p_ = 100. * p_;
  q_.setIdentity();
  r_.setIdentity();

  this->feetHeights.setZero(4);
}


void KalmanFilterEstimate::ResetStateEst()
{
  // std::cout << "############ reset state estimation ########## " << std::endl;

  // this->pSCg_z_inv_last.resize(this->robot_info.generalizedCoordinatesNum);
  // this->pSCg_z_inv_last.setZero();

  // this->cmd_torque.resize(this->robot_info.actuatedDofNum);
  // this->cmd_torque.setZero();
  
  // this->kalmanMatrix.xHat.setZero();
  
  // this->kalmanMatrix.foot_positions.setZero();
  // this->kalmanMatrix.foot_velocities.setZero();

  // this->kalmanMatrix.A_matrix.setZero();
  // this->kalmanMatrix.A_matrix.block(0, 0, 3, 3)   = Eigen::Matrix<scalar_t, 3, 3>::Identity();
  // this->kalmanMatrix.A_matrix.block(3, 3, 3, 3)   = Eigen::Matrix<scalar_t, 3, 3>::Identity();
  // this->kalmanMatrix.A_matrix.block(6, 6, 12, 12) = Eigen::Matrix<scalar_t, 12, 12>::Identity();

  // this->kalmanMatrix.B_matrix.setZero();

  // Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> c1(3, 6);
  // c1 << Eigen::Matrix<scalar_t, 3, 3>::Identity(), Eigen::Matrix<scalar_t, 3, 3>::Zero();
  // Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> c2(3, 6);
  // c2 << Eigen::Matrix<scalar_t, 3, 3>::Zero(), Eigen::Matrix<scalar_t, 3, 3>::Identity();

  // this->kalmanMatrix.C_matrix.setZero();
  // this->kalmanMatrix.C_matrix.block(0, 0, 3, 6) = c1;
  // this->kalmanMatrix.C_matrix.block(3, 0, 3, 6) = c1;
  // this->kalmanMatrix.C_matrix.block(6, 0, 3, 6) = c1;
  // this->kalmanMatrix.C_matrix.block(9, 0, 3, 6) = c1;
  // this->kalmanMatrix.C_matrix.block(0, 6, 12, 12) = -Eigen::Matrix<scalar_t, 12, 12>::Identity();
  // this->kalmanMatrix.C_matrix.block(12, 0, 3, 6) = c2;
  // this->kalmanMatrix.C_matrix.block(15, 0, 3, 6) = c2;
  // this->kalmanMatrix.C_matrix.block(18, 0, 3, 6) = c2;
  // this->kalmanMatrix.C_matrix.block(21, 0, 3, 6) = c2;
  // this->kalmanMatrix.C_matrix(27, 17) = 1.0;
  // this->kalmanMatrix.C_matrix(26, 14) = 1.0;
  // this->kalmanMatrix.C_matrix(25, 11) = 1.0;
  // this->kalmanMatrix.C_matrix(24, 8)  = 1.0;

  // this->kalmanMatrix.P_posteriori_covariance_matrix.setIdentity();
  // this->kalmanMatrix.P_posteriori_covariance_matrix = 100. * this->kalmanMatrix.P_posteriori_covariance_matrix;
  // this->kalmanMatrix.Q_obs_process_noise_covariance_matrix.setIdentity();
  // this->kalmanMatrix.R_state_process_noise_covariance_matrix.setIdentity();

  // this->feetHeights.setZero(4);

  // this->lin_vel_filtered.setZero();

  // this->contact_tick[0] = 0;
  // this->contact_tick[1] = 0;
  // this->contact_tick[2] = 0;
  // this->contact_tick[3] = 0;
}




template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void KalmanFilterEstimate::ReadKalmanYaml(std::string robot_path)
{
    // The config file is located at "rl_sar/src/rl_sar/policy/<robot_path>/config.yaml"
    std::string config_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/config/kalman_config/" + robot_path + "/kalman_config.yaml";
    std::cout << "[Loading] kalman config from " << config_path << std::endl;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[robot_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    this->robot_info.joint_names = ReadVectorFromYaml<std::string>(config["joint_names"]);
    this->robot_info.foot_names = ReadVectorFromYaml<std::string>(config["foot_names"]);

    this->robot_info.generalizedCoordinatesNum = config["generalizedCoordinatesNum"].as<int>();
    this->robot_info.actuatedDofNum = config["actuatedDofNum"].as<int>();
    this->robot_info.cutoff_frequency = config["cutoff_frequency"].as<scalar_t>();
    this->robot_info.contact_threshold = config["contact_threshold"].as<scalar_t>();
    this->noise_settings.footRadius = config["footRadius"].as<scalar_t>();
    this->noise_settings.imuProcessNoisePosition = config["imuProcessNoisePosition"].as<scalar_t>();
    this->noise_settings.imuProcessNoiseVelocity = config["imuProcessNoiseVelocity"].as<scalar_t>();
    this->noise_settings.footProcessNoisePosition = config["footProcessNoisePosition"].as<scalar_t>();
    this->noise_settings.footSensorNoisePosition = config["footSensorNoisePosition"].as<scalar_t>();
    this->noise_settings.footSensorNoiseVelocity = config["footSensorNoiseVelocity"].as<scalar_t>();
    this->noise_settings.footHeightSensorNoise = config["footHeightSensorNoise"].as<scalar_t>();



}


vector_t KalmanFilterEstimate::Update(const scalar_t& dt, RobotState<double> *state) {

  if(this->first_update){
    this->first_update = false;
  }
  else{
    estContactForce(dt);
  }

  quaternion_t quat;
  quat.w() = state->imu.quaternion[0];
  quat.x() = state->imu.quaternion[1];
  quat.y() = state->imu.quaternion[2];
  quat.z() = state->imu.quaternion[3];

  vector3_t zyx = quatToZyx(quat);

  vector3_t angularVelLocal;
  for (size_t i = 0; i < 3; ++i) {
    angularVelLocal[i] = state->imu.gyroscope[i];
  }
  vector3_t angularVelGlobal = getGlobalAngularVelocityFromEulerAnglesZyxDerivatives<scalar_t>(
      zyx, getEulerAnglesZyxDerivativesFromLocalAngularVelocity<scalar_t>(zyx, angularVelLocal));

  for (size_t i = 0; i < this->robot_info.actuatedDofNum; ++i) {
    this->rbdState[12 + i] = state->motor_state.q[i];
    this->rbdState[12 + this->robot_info.actuatedDofNum + i] = state->motor_state.dq[i];
    this->cmd_torque[i] = state->motor_state.tau_est[i];
  }

  this->rbdState.segment<3>(3) = zyx;
  this->rbdState.segment<3>(9) = angularVelGlobal;

  a_.block(0, 3, 3, 3) = dt * Eigen::Matrix<scalar_t, 3, 3>::Identity();
  b_.block(0, 0, 3, 3) = 0.5 * dt * dt * Eigen::Matrix<scalar_t, 3, 3>::Identity();
  b_.block(3, 0, 3, 3) = dt * Eigen::Matrix<scalar_t, 3, 3>::Identity();
  q_.block(0, 0, 3, 3) = (dt / 20.f) * Eigen::Matrix<scalar_t, 3, 3>::Identity();
  q_.block(3, 3, 3, 3) = (dt * 9.81f / 20.f) * Eigen::Matrix<scalar_t, 3, 3>::Identity();
  q_.block(6, 6, 12, 12) = dt * Eigen::Matrix<scalar_t, 12, 12>::Identity();

  const auto& model = this->pinocchioInterfacePtr->getModel();
  auto& data = this->pinocchioInterfacePtr->getData();
  size_t actuatedDofNum = this->robot_info.actuatedDofNum;

  vector_t qPino(this->robot_info.generalizedCoordinatesNum);
  vector_t vPino(this->robot_info.generalizedCoordinatesNum);
  qPino.setZero();
  qPino.segment<3>(3) = this->rbdState.segment(3,3);  // Only set orientation, let position in origin.
  qPino.tail(actuatedDofNum) = this->rbdState.segment(12, actuatedDofNum);

  vPino.setZero();
  vPino.segment<3>(3) = getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(
      qPino.segment<3>(3),
       this->rbdState.segment<3>(9));  // Only set angular velocity, let linear velocity be zero
  vPino.tail(actuatedDofNum) = this->rbdState.segment(12 + actuatedDofNum , actuatedDofNum);

  pinocchio::forwardKinematics(model, data, qPino, vPino);
  pinocchio::updateFramePlacements(model, data);

  // TODO left_foot_front_link, right_foot_front_link, left_foot_hind_link, right_foot_hind_link
  const auto eePos_4 = this->eeKinematicsPtr->getPosition(vector_t());
  const auto eeVel_4 = this->eeKinematicsPtr->getVelocity(vector_t(), vector_t());

  std::vector<vector3_t> eePos, eeVel;

  for (size_t i = 0; i < 4; ++i) {
    eePos.emplace_back(eePos_4[i]); 
    eeVel.emplace_back(eeVel_4[i]); 
  }

  // TODO left_foot_front_link, right_foot_front_link, left_foot_hind_link, right_foot_hind_link

  if (this->contactFlag[0]) // left
  {
    if (eePos[0][2] < eePos[2][2] - this->noise_settings.footRadius) // 前脚低于后脚
      contactFlag[2] = false;
    else if (eePos[2][2] < eePos[0][2] - this->noise_settings.footRadius)
      contactFlag[0] = false;
  }
  if (this->contactFlag[1]) // right
  {
    if (eePos[1][2] < eePos[3][2] - this->noise_settings.footRadius) 
      contactFlag[3] = false;
    else if (eePos[3][2] < eePos[1][2] - this->noise_settings.footRadius)
      contactFlag[1] = false;
  }

// //判断是否是可靠接触状态
//   for (int i = 0; i < 4; i++)
//   {
//     if(this->contactFlag[i]){
//       this->contact_tick[i] += 1; 
//     }else{
//       this->contact_tick[i] = 0;
//     }
//     if(this->contact_tick[i] >= 20){ //接触满100ms以后，则认为是可靠的接触状态
//       this->contactFlag_reliable[i] = 1;
//     }else{
//       this->contactFlag_reliable[i] = 0;
//     }
//   }

  // the covariance of the process noise
  Eigen::Matrix<scalar_t, 18, 18> q = Eigen::Matrix<scalar_t, 18, 18>::Identity();
  q.block(0, 0, 3, 3) = q_.block(0, 0, 3, 3) * this->noise_settings.imuProcessNoisePosition;
  q.block(3, 3, 3, 3) = q_.block(3, 3, 3, 3) * this->noise_settings.imuProcessNoiseVelocity;
  q.block(6, 6, 12, 12) = q_.block(6, 6, 12, 12) * this->noise_settings.footProcessNoisePosition;

  // the covariance of the observation noise
  Eigen::Matrix<scalar_t, 28, 28> r = Eigen::Matrix<scalar_t, 28, 28>::Identity();
  r.block(0, 0, 12, 12) = r_.block(0, 0, 12, 12) * this->noise_settings.footSensorNoisePosition;
  r.block(12, 12, 12, 12) = r_.block(12, 12, 12, 12) * this->noise_settings.footSensorNoiseVelocity;
  const int fn = 4;
  r.block(24, 24, fn, fn) = r_.block(24, 24, fn, fn) * this->noise_settings.footHeightSensorNoise;

  // for (int i = 0; i < this->robot_info.numThreeDofContacts; i++) {
  for (int i = 0; i < 4; i++) {
    int i1 = 3 * i;

    int qIndex = 6 + i1;
    int rIndex1 = i1;
    int rIndex2 = 12 + i1;
    int rIndex3 = 24 + i;
    bool isContact = contactFlag[i];
    // bool isContact = contactFlag_reliable[i];

    scalar_t high_suspect_number(1000);
    q.block(qIndex, qIndex, 3, 3) = (isContact ? 1. : high_suspect_number) * q.block(qIndex, qIndex, 3, 3);
    r.block(rIndex1, rIndex1, 3, 3) = (isContact ? 1. : high_suspect_number) * r.block(rIndex1, rIndex1, 3, 3);
    r.block(rIndex2, rIndex2, 3, 3) = (isContact ? 1. : high_suspect_number) * r.block(rIndex2, rIndex2, 3, 3);
    r(rIndex3, rIndex3) = (isContact ? 1. : high_suspect_number) * r(rIndex3, rIndex3);

    ps_.segment(3 * i, 3) = -eePos[i];
    ps_.segment(3 * i, 3)[2] += this->noise_settings.footRadius;
    vs_.segment(3 * i, 3) = -eeVel[i];
  }
  // printf("body foot height 0: %f, 1: %f, 2: %f, 3: %f\n", ps_(2), ps_(5), ps_(8), ps_(11));

  vector3_t linearAccelLocal;
  for (size_t i = 0; i < 3; ++i) {
    linearAccelLocal[i] = state->imu.accelerometer[i];
  }

  vector3_t g(0, 0, -9.81);
  vector3_t accel = getRotationMatrixFromZyxEulerAngles(zyx) * linearAccelLocal + g;

  // observation (or measurement)
  Eigen::Matrix<scalar_t, 28, 1> y;
  // syx added ref: https://en.wikipedia.org/wiki/Kalman_filter#Details
  // a_=F c_=H
  y << ps_, vs_, this->feetHeights;  

  // this->kalmanMatrix.xHat = this->kalmanMatrix.A_matrix * this->kalmanMatrix.xHat + this->kalmanMatrix.B_matrix * accel;  // Predicted (a priori) state estimate
  xHat_ = a_ * xHat_ + b_ * accel;

  // Eigen::Matrix<scalar_t, 12, 1> vf;//表示的是足端接触点的线速度

  // vector3_t base_lin_vel = this->kalmanMatrix.xHat.segment<3>(3);
  // // 使用 replicate 方法重复块
  // vf = base_lin_vel.replicate(4, 1).eval();

  // vf = vf - this->kalmanMatrix.foot_velocities;

  // //根据估计的足端速度判断是否相信足端零速度模型
  // for (int i = 0; i < 4; i++)
  // {
  //   int i1 = 3 * i;

  //   int qIndex = 6 + i1;
  //   int rIndex1 = i1;
  //   int rIndex2 = 12 + i1;
  //   int rIndex3 = 24 + i;
  //   bool isContact = this->contactFlag[i];//原先是contactFlag
  //   // bool isContact = contactFlag_reliable[i];

  //   if(isContact){
  //     if(vf.block(3*i, 0, 3, 1).norm() > 1.0){//此时认为发生了滑动，需要将方差增大
  //       scalar_t high_suspect_number(1000);
  //       q.block(qIndex, qIndex, 3, 3) = high_suspect_number * q.block(qIndex, qIndex, 3, 3);
  //       r.block(rIndex1, rIndex1, 3, 3) = high_suspect_number * r.block(rIndex1, rIndex1, 3, 3);
  //       r.block(rIndex2, rIndex2, 3, 3) = high_suspect_number * r.block(rIndex2, rIndex2, 3, 3);
  //       r(rIndex3, rIndex3) = high_suspect_number * r(rIndex3, rIndex3);
  //     }
  //   }
  // } 
  
  Eigen::Matrix<scalar_t, 18, 18> at = a_.transpose();
  Eigen::Matrix<scalar_t, 18, 18> pm = a_ * p_ * at + q;  
  Eigen::Matrix<scalar_t, 18, 28> cT = c_.transpose();
  Eigen::Matrix<scalar_t, 28, 1> yModel = c_ * xHat_;
  Eigen::Matrix<scalar_t, 28, 1> ey = y - yModel;        
  Eigen::Matrix<scalar_t, 28, 28> s = c_ * pm * cT + r;  
  Eigen::Matrix<scalar_t, 28, 1> sEy = s.lu().solve(ey);  // *Roughly* Optimal Kalman gain
  xHat_ += pm * cT * sEy;                                 

  Eigen::Matrix<scalar_t, 28, 18> sC = s.lu().solve(c_);
  p_ = (Eigen::Matrix<scalar_t, 18, 18>::Identity() - pm * cT * sC) *
       pm;  

  Eigen::Matrix<scalar_t, 18, 18> pt = p_.transpose();
  p_ = (p_ + pt) / 2.0;

  if (p_.block(0, 0, 2, 2).determinant() > 0.000001)
  {
    p_.block(0, 2, 2, 16).setZero();
    p_.block(2, 0, 16, 2).setZero();
    p_.block(0, 0, 2, 2) /= 10.;
  }

  // if (SWING_TEST)
  // {
  //   xHat_.head<6>().setZero();
  //   xHat_(2) = 0.6;
  // }
  // ROS_WARN_STREAM("xHat_dddddd: " << xHat_.transpose());
  // 检查每个值是否在合法范围内
  // if ((xHat_.array() > MAX_VALUE).any() || (xHat_.array() < MIN_VALUE).any()) 
  // {

  //       throw OutOfRangeException(); // 抛出自定义异常
  // }
  
  this->lin_vel_filtered = 0.8*this->lin_vel_filtered + (1-0.8)*xHat_.segment<3>(3);

  this->rbdState.segment<3>(0) = xHat_.segment<3>(0);
  this->rbdState.segment<3>(6) = this->lin_vel_filtered;

  return this->rbdState;
}

void KalmanFilterEstimate::estContactForce(const scalar_t& dt)
{
  // if (dt > 1) dt = 0.005; // 0.002
  const scalar_t lamda = this->robot_info.cutoff_frequency;
  const scalar_t gama = exp(-lamda * dt);
  const scalar_t beta = (1 - gama) / (gama * dt);

  auto qMeasured = vector_t(this->robot_info.generalizedCoordinatesNum);
  auto vMeasured = vector_t(this->robot_info.generalizedCoordinatesNum);
  const auto& tauCmd = this->cmd_torque;


  qMeasured.head<6>() = this->rbdState.head<6>() ;
  qMeasured.tail(this->robot_info.actuatedDofNum) = this->rbdState.segment(12, this->robot_info.actuatedDofNum);

  vMeasured.head<3>() = this->rbdState.segment<3>(6);
  vMeasured.segment<3>(3) = getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(
      qMeasured.segment<3>(3), this->rbdState.segment<3>(9));
  vMeasured.tail(this->robot_info.actuatedDofNum) = this->rbdState.segment(this->robot_info.generalizedCoordinatesNum + 6, this->robot_info.actuatedDofNum);

  const auto& model = this->pinocchioInterfacePtr->getModel();
  auto& data = this->pinocchioInterfacePtr->getData();

  matrix_t s(this->robot_info.actuatedDofNum, this->robot_info.generalizedCoordinatesNum);
  s.block(0, 0, this->robot_info.actuatedDofNum, 6).setZero();
  s.block(0, 6, this->robot_info.actuatedDofNum, this->robot_info.actuatedDofNum).setIdentity();

  pinocchio::forwardKinematics(model, data, qMeasured, vMeasured);
  pinocchio::computeJointJacobians(model, data);
  pinocchio::updateFramePlacements(model, data);

  pinocchio::crba(model, data, qMeasured);
  data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();

  pinocchio::getCoriolisMatrix(model, data);
  // data.C;

  pinocchio::computeGeneralizedGravity(model, data, qMeasured);
  // data.g;

  vector_t p = data.M * vMeasured;

  vector_t pSCg = beta * p + s.transpose() * tauCmd + data.C.transpose() * vMeasured - data.g;

  vector_t pSCg_z_inv = (1 - gama) * pSCg + gama * this->pSCg_z_inv_last;
  this->pSCg_z_inv_last = pSCg_z_inv;

  this->est_disturbance_torque = beta * p - pSCg_z_inv;

  const std::vector<size_t> center_foot_id = {model.getBodyId("left_ankle_roll_link"), model.getBodyId("right_ankle_roll_link")};

  auto Jac_i = matrix_t(6, this->robot_info.generalizedCoordinatesNum);
  auto S_li = matrix_t(6, this->robot_info.generalizedCoordinatesNum);
  for (size_t i = 0; i < 2; ++i) {
    Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    jac.setZero(6, this->robot_info.generalizedCoordinatesNum);

    pinocchio::getFrameJacobian(model, data, center_foot_id[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);

    Jac_i = jac.template topRows<6>();
    S_li.setZero();
    int index = 0;
    if (i == 0)
      index = 0;
    else if (i == 1)
      index = 6;
    S_li.block<6, 6>(0, 6+index) = Eigen::Matrix<scalar_t, 6, 6>::Identity();
    this->est_contact_force.segment<6>(6*i) = (S_li*Jac_i.transpose()).inverse() * S_li * this->est_disturbance_torque;
  //   matrix6_t S_JT = S_li * Jac_i.transpose();
  //   vector6_t S_tau = S_li * this->est_disturbance_torque;
  //   this->est_contact_force.segment<6>(6*i) = S_JT.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(S_tau);
  }

  // left force(6) + right force(6) + left linear norm(1) + right linear norm(1) + left linear angular norm(1) + right linear angular norm(1)
  for (int i = 0; i < 2; i++)
  {
    this->est_contact_force(6*2 + i) = this->est_contact_force.segment<3>(6*i).norm();
  }
  for (int i = 0; i < 2; i++)
  {
    this->est_contact_force(6*2 + 2 + i) = this->est_contact_force.segment<6>(6*i).norm();
  }
  // std::cout << "est contact force:" << this->est_contact_force.transpose() << std::endl;

  // std::cout << "est contact force  left: " << this->est_contact_force[2] << std::endl;
  // std::cout << "est contact force  right: " << this->est_contact_force[8] << std::endl;

  if(this->est_contact_force[2] > this->robot_info.contact_threshold){
    this->contactFlag[0] = true;
    this->contactFlag[2] = true;
    // std::cout << std::flush  << " left contact "<< std::flush;
  }
  else{
    this->contactFlag[0] = false;
    this->contactFlag[2] = false;
  }
  if(this->est_contact_force[8] > this->robot_info.contact_threshold){
    this->contactFlag[1] = true;
    this->contactFlag[3] = true;
    // std::cout << std::flush  << " right contact "<< std::flush;
  }
  else{
    this->contactFlag[1] = false;
    this->contactFlag[3] = false;
  }


}


/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioInterface KalmanFilterEstimate::createPinocchioInterface(const std::string& urdfFilePath, const std::vector<std::string>& jointNames) {
  using joint_pair_t = std::pair<const std::string, std::shared_ptr<::urdf::Joint>>;

  ::urdf::ModelInterfaceSharedPtr urdfTree = ::urdf::parseURDFFile(urdfFilePath);
  if (urdfTree == nullptr) {
    throw std::invalid_argument("The file " + urdfFilePath + " does not contain a valid URDF model!");
  }

  // remove extraneous joints from urdf
  ::urdf::ModelInterfaceSharedPtr newModel = std::make_shared<::urdf::ModelInterface>(*urdfTree);
  for (joint_pair_t& jointPair : newModel->joints_) {
    if (std::find(jointNames.begin(), jointNames.end(), jointPair.first) == jointNames.end()) {
      jointPair.second->type = urdf::Joint::FIXED;
    }
  }

  // add 6 DoF for the floating base
  pinocchio::JointModelComposite jointComposite(2);
  jointComposite.addJoint(pinocchio::JointModelTranslation());
  jointComposite.addJoint(pinocchio::JointModelSphericalZYX());

  return getPinocchioInterfaceFromUrdfModel(newModel, jointComposite);
}




}  // namespace legged
