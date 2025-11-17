#include "math_rl.hpp"

// 将欧拉角转换为四元数
Eigen::Quaterniond euler_single_axis_to_quat(double angle, char axis, bool degrees) {
    if (degrees) {
        angle = angle * M_PI / 180.0;
    }
    Eigen::Quaterniond quat;
    switch (axis) {
        case 'x':
            quat = Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()));
            break;
        case 'y':
            quat = Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));
            break;
        case 'z':
            quat = Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()));
            break;
        default:
            throw std::invalid_argument("Axis must be 'x', 'y', or 'z'");
    }
    // 归一化四元数
    quat.normalize();
    return quat;
}


Eigen::Matrix3d matrix_from_quat(const Eigen::Quaterniond& q) {
    return q.toRotationMatrix();
}

vector3_t QuatRotateInverse(const quaternion_t& q, const vector3_t& v)
{
    // 提取四元数分量 (w, x, y, z)
    double q_w = q.w();
    vector3_t q_vec(q.x(), q.y(), q.z()); // 正确提取四元数的向量部分

    // 计算旋转后的向量
    vector3_t result = v * (2.0 * q_w*q_w - 1.0);
    result -= 2.0 * q_w * q_vec.cross(v);
    result += 2.0 * q_vec * q_vec.dot(v);

    return result;
}

Eigen::Quaterniond yaw_quat(const Eigen::Quaterniond& q) {
    double yaw = atan2(2.0 * (q.w() * q.z() + q.x() * q.y()), 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
    Eigen::Quaterniond quat;
    quat = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    return quat;
}

