#include "atom/common/bridge.h"
#include "atom/common/timer.h"

#include <fstream>
#include <iostream>

int main(int argc, char *argv[])
{
    // set cyclonedds config file path
#ifdef _SET_CYCLONEDDS_URI
    const char *config_file = "../config/cyclonedds.xml";
    if (setenv("CYCLONEDDS_URI", config_file, 1) != 0) {
        std::cout << "setenv CYCLONEDDS_URI failed" << std::endl;
    } else {
        std::cout << "CYCLONEDDS_URI set to config_file: " << config_file << std::endl;
        std::cout << "请查看配置文件说明，确保配置文件中的网络接口名称与连接机器人的网卡名称一致。" << std::endl;
    }
#endif

    Atom::Bridge bridge;
    std::ofstream file("joint_angles.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open joint_angles.txt" << std::endl;
        return 1;
    }
    // time unit: s
    const float init_duration = 5.0;
    const float end_duration = 20.0;
    const float control_dt = 0.01;
    const int s2us = 1e6;
    double time = 0.0;
    Timer timer;
    std::vector<Eigen::VectorXf> q_ref;
    constexpr int kRecordDofs = Atom::kLegDofs + Atom::kArmDofs;

    while (1) {
        const std::shared_ptr<const Atom::ArmJointState> ajs_tmp_ptr = bridge.GetNewestArmStatePtr();
        const std::shared_ptr<const Atom::JointState> js_tmp_ptr = bridge.GetNewestJointStatePtr();
        if (nullptr == ajs_tmp_ptr || nullptr == js_tmp_ptr) {
            std::cerr << "Failed to get joint state ptr" << std::endl;
            return 1;
        }
        timer.Tic();
        time += control_dt;
        if (time > init_duration && time < end_duration) {
            std::cout << "start time: " << init_duration << ", end time: " << end_duration << ", current time: " << time << std::endl;
            Eigen::VectorXf q_current(kRecordDofs);
            for (size_t i = 0; i < kRecordDofs; ++i) {
                if (i < Atom::kLegDofs) {
                    q_current[i] = js_tmp_ptr->q[i];
                } else {
                    q_current[i] = ajs_tmp_ptr->q[i - Atom::kLegDofs];
                }
                file << q_current[i];
                if (i < kRecordDofs) file << " ";
            }
            q_ref.push_back(q_current);
            file << "\n";
        }
        timer.Toc();
        timer.usDelay(control_dt * s2us - timer.usDuration());
        if (time > end_duration) { break; }
    }

    for (auto it = q_ref.rbegin(); it != q_ref.rend(); ++it) {
        for (size_t i = 0; i < kRecordDofs; ++i) {
            file << (*it)[i];
            if (i < kRecordDofs) file << " ";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Done ==========" << std::endl;
    return 0;
}