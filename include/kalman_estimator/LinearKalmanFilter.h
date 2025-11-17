//

#ifndef E625DC3F_4CE4_4BDB_B1EB_D69CD4FEB9ED
#define E625DC3F_4CE4_4BDB_B1EB_D69CD4FEB9ED

#include "rl_sdk.hpp"
#include "PinocchioInterface.h"
#include "PinocchioEndEffectorKinematics.h"
#include "urdf.h"
#include "math_rl.hpp"

#include <stdexcept>

class OutOfRangeException : public std::exception {
public:
    const char* what() const noexcept override {
        return "Error: Value out of range!";
    }
};

namespace legged {

using namespace ocs2;

struct KalmanMatrix
{
    Eigen::Matrix<scalar_t, 18, 1> xHat;  // the a posteriori state estimate mean
    Eigen::Matrix<scalar_t, 12, 1> foot_positions;
    Eigen::Matrix<scalar_t, 12, 1> foot_velocities;
    Eigen::Matrix<scalar_t, 18, 18> A_matrix;  // the state-transition model
    Eigen::Matrix<scalar_t, 18, 18> Q_obs_process_noise_covariance_matrix;  // the covariance of the process noise
    Eigen::Matrix<scalar_t, 18, 18> P_posteriori_covariance_matrix;  // the a posteriori estimate covariance matrix
    Eigen::Matrix<scalar_t, 28, 28> R_state_process_noise_covariance_matrix;
    Eigen::Matrix<scalar_t, 18, 3>  B_matrix;  // the control-input model
    Eigen::Matrix<scalar_t, 28, 18> C_matrix;  // the observation model
};

struct KalmanNoiseSettings
{
    scalar_t footRadius = 0.005;
    scalar_t imuProcessNoisePosition = 0.02;
    scalar_t imuProcessNoiseVelocity = 0.02;
    scalar_t footProcessNoisePosition = 0.002;
    scalar_t footSensorNoisePosition = 0.005;
    scalar_t footSensorNoiseVelocity = 0.1;
    scalar_t footHeightSensorNoise = 0.01;
};



struct RobotInfo
{
    int generalizedCoordinatesNum;
    int actuatedDofNum;
    std::vector<std::string> joint_names;
    std::vector<std::string> foot_names;
    scalar_t cutoff_frequency = 150;
    scalar_t contact_threshold = 23;
};




class KalmanFilterEstimate {
    public:
    KalmanFilterEstimate(std::string robot_name);

    vector_t Update(const scalar_t& dt, RobotState<double> *state);
    vector_t UpdateRobotState(RobotState<double> *state);
    void ReadKalmanYaml(std::string robot_path);
    void setCmdTorque(const vector_t& torque) { this->cmd_torque = torque; }
    void updateContact(vector_t desired_contact)
    {
        if(desired_contact[0] > 0.9){
            this->contactFlag[0] = this->contactFlag[2] = true;
        }
        else{
            this->contactFlag[0] = this->contactFlag[2] = false;
        }
        if(desired_contact[1] > 0.9){
            this->contactFlag[1] = this->contactFlag[3] = true;
        }
        else{
            this->contactFlag[1] = this->contactFlag[3] = false;
        }
    }
    void estContactForce(const scalar_t& dt);
    const vector_t& getEstContactForce()
    {
        return this->est_contact_force;
    }    
    
    void ResetStateEst();
    
    /**
     * Create a CentroidalModel PinocchioInterface from a URDF.
     * @param [in] urdfFilePath: The absolute path to the URDF file for the robot.
     * @param [in] jointNames: Any joint that is not listed in jointNames (a.k.a the extraneous joints) will be removed from the urdf.
     */
    PinocchioInterface createPinocchioInterface(const std::string& urdfFilePath, const std::vector<std::string>& jointNames);

    protected:

    uint64_t contact_tick[4];//计数表示接触的时间，若为腾空状态，则清零，若，contactFlag_=1,则该变量不断的+1，目前的代码为计数50(100ms)
                           //则contactFlag_use_to_compute{}置1，
    std::array<bool, 4> contactFlag{true, true, true, true};
    std::array<bool, 4> contactFlag_reliable{true, true, true, true};//表示的是实际用于状态估计的接触状态(可靠接触状态)，该状态的目的是为了消除接触瞬间冲击的影响

    bool first_update = true;

    vector_t rbdState;
    vector_t feetHeights;

    vector_t est_contact_force;
    vector_t est_disturbance_torque;
    vector_t pSCg_z_inv_last;
    vector_t cmd_torque;
    std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr;
    std::shared_ptr<PinocchioEndEffectorKinematics> eeKinematicsPtr;

    vector3_t lin_vel_filtered; 

    // Config
    KalmanNoiseSettings noise_settings;
    RobotInfo robot_info;

    private:

    // KalmanMatrix kalmanMatrix;
    Eigen::Matrix<scalar_t, 18, 1> xHat_;
    Eigen::Matrix<scalar_t, 12, 1> ps_;
    Eigen::Matrix<scalar_t, 12, 1> vs_;
    Eigen::Matrix<scalar_t, 18, 18> a_;
    Eigen::Matrix<scalar_t, 18, 18> q_;
    Eigen::Matrix<scalar_t, 18, 18> p_;
    Eigen::Matrix<scalar_t, 28, 28> r_;
    Eigen::Matrix<scalar_t, 18, 3> b_;
    Eigen::Matrix<scalar_t, 28, 18> c_;
    // const double MAX_VALUE = 1.0e+10;
    // const double MIN_VALUE = -1.0e+10;

};

}  // namespace legged


#endif /* E625DC3F_4CE4_4BDB_B1EB_D69CD4FEB9ED */
