#ifndef F7E9056B_8FE2_4437_89C8_E07473EA4A18
#define F7E9056B_8FE2_4437_89C8_E07473EA4A18

#include <Eigen/Dense>

using scalar_t = double;
using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
using matrix3_t = Eigen::Matrix<scalar_t, 3, 3>;
using quaternion_t = Eigen::Quaternion<scalar_t>;
using tensor_element_t = float;


Eigen::Quaterniond euler_single_axis_to_quat(double angle, char axis, bool degrees = false);
Eigen::Matrix3d matrix_from_quat(const Eigen::Quaterniond& quat);
vector3_t QuatRotateInverse(const quaternion_t& q, const vector3_t& v);
Eigen::Quaterniond yaw_quat(const Eigen::Quaterniond& quat);


template <typename T>
T square(T a) {
  return a * a;
}

template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 1> quatToZyx(const Eigen::Quaternion<SCALAR_T>& q) {
    Eigen::Matrix<SCALAR_T, 3, 1> zyx;

    SCALAR_T as = std::min(-2. * (q.x() * q.z() - q.w() * q.y()), .99999);
    zyx(0) = std::atan2(2 * (q.x() * q.y() + q.w() * q.z()), square(q.w()) + square(q.x()) - square(q.y()) - square(q.z()));
    zyx(1) = std::asin(as);
    zyx(2) = std::atan2(2 * (q.y() * q.z() + q.w() * q.x()), square(q.w()) - square(q.x()) - square(q.y()) + square(q.z()));
    return zyx;
}

/**
 * Compute angular velocities expressed in the world frame from derivatives of ZYX-Euler angles
 *
 * @param [in] eulerAngles: ZYX-Euler angles
 * @param [in] derivativesEulerAngles: time-derivative of ZYX-Euler angles
 * @return angular velocity expressed in world frame
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 1> getGlobalAngularVelocityFromEulerAnglesZyxDerivatives(
    const Eigen::Matrix<SCALAR_T, 3, 1>& eulerAngles, const Eigen::Matrix<SCALAR_T, 3, 1>& derivativesEulerAngles) {
  const SCALAR_T sz = sin(eulerAngles(0));
  const SCALAR_T cz = cos(eulerAngles(0));
  const SCALAR_T sy = sin(eulerAngles(1));
  const SCALAR_T cy = cos(eulerAngles(1));
  const SCALAR_T dz = derivativesEulerAngles(0);
  const SCALAR_T dy = derivativesEulerAngles(1);
  const SCALAR_T dx = derivativesEulerAngles(2);
  return {-sz * dy + cy * cz * dx, cz * dy + cy * sz * dx, dz - sy * dx};
}


/**
 * Compute derivatives of ZYX-Euler angles from local angular velocities
 * The inverse of getLocalAngularVelocityFromEulerAnglesZyxDerivatives
 *
 * This transformation is singular for y = +- pi / 2
 *
 * @param [in] eulerAngles: ZYX-Euler angles
 * @param [in] angularVelocity: angular velocity expressed in local frame
 * @return derivatives of ZYX-Euler angles
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 1> getEulerAnglesZyxDerivativesFromLocalAngularVelocity(const Eigen::Matrix<SCALAR_T, 3, 1>& eulerAngles,
                                                                                   const Eigen::Matrix<SCALAR_T, 3, 1>& angularVelocity) {
  const SCALAR_T sy = sin(eulerAngles(1));
  const SCALAR_T cy = cos(eulerAngles(1));
  const SCALAR_T sx = sin(eulerAngles(2));
  const SCALAR_T cx = cos(eulerAngles(2));
  const SCALAR_T wx = angularVelocity(0);
  const SCALAR_T wy = angularVelocity(1);
  const SCALAR_T wz = angularVelocity(2);
  const SCALAR_T tmp = sx * wy / cy + cx * wz / cy;
  return {tmp, cx * wy - sx * wz, wx + sy * tmp};
}

/**
 * Compute derivatives of ZYX-Euler angles from global angular velocities
 * The inverse of getGlobalAngularVelocityFromEulerAnglesZyxDerivatives
 *
 * This transformation is singular for y = +- pi / 2
 *
 * @param [in] eulerAngles: ZYX-Euler angles
 * @param [in] angularVelocity: angular velocity expressed in world frame
 * @return derivatives of ZYX-Euler angles
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 1> getEulerAnglesZyxDerivativesFromGlobalAngularVelocity(const Eigen::Matrix<SCALAR_T, 3, 1>& eulerAngles,
                                                                                    const Eigen::Matrix<SCALAR_T, 3, 1>& angularVelocity) {
  const SCALAR_T sz = sin(eulerAngles(0));
  const SCALAR_T cz = cos(eulerAngles(0));
  const SCALAR_T sy = sin(eulerAngles(1));
  const SCALAR_T cy = cos(eulerAngles(1));
  const SCALAR_T wx = angularVelocity(0);
  const SCALAR_T wy = angularVelocity(1);
  const SCALAR_T wz = angularVelocity(2);
  const SCALAR_T tmp = cz * wx / cy + sz * wy / cy;
  return {sy * tmp + wz, -sz * wx + cz * wy, tmp};
}


/**
 * Compute the rotation matrix corresponding to euler angles zyx
 *
 * @param [in] eulerAnglesZyx
 * @return The corresponding rotation matrix
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 3> getRotationMatrixFromZyxEulerAngles(const Eigen::Matrix<SCALAR_T, 3, 1>& eulerAngles) {
  const SCALAR_T z = eulerAngles(0);
  const SCALAR_T y = eulerAngles(1);
  const SCALAR_T x = eulerAngles(2);

  const SCALAR_T c1 = cos(z);
  const SCALAR_T c2 = cos(y);
  const SCALAR_T c3 = cos(x);
  const SCALAR_T s1 = sin(z);
  const SCALAR_T s2 = sin(y);
  const SCALAR_T s3 = sin(x);

  const SCALAR_T s2s3 = s2 * s3;
  const SCALAR_T s2c3 = s2 * c3;

  // clang-format off
  Eigen::Matrix<SCALAR_T, 3, 3> rotationMatrix;
  rotationMatrix << c1 * c2,      c1 * s2s3 - s1 * c3,       c1 * s2c3 + s1 * s3,
                    s1 * c2,      s1 * s2s3 + c1 * c3,       s1 * s2c3 - c1 * s3,
                        -s2,                  c2 * s3,                   c2 * c3;
  // clang-format on
  return rotationMatrix;
}

template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 1> quatToXyz(const Eigen::Quaternion<SCALAR_T>& q) {
  Eigen::Matrix<SCALAR_T, 3, 1> xyz;

  // Roll (X-axis rotation)
    SCALAR_T sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
    SCALAR_T cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
    xyz(0) = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (Y-axis rotation)
    SCALAR_T sinp = 2 * (q.w() * q.y() - q.z() * q.x());
    if (std::abs(sinp) >= 1)
        xyz(1) = std::copysign(M_PI / 2, sinp); // 使用copysign来处理极端情况
    else
        xyz(1) = std::asin(sinp);

    // Yaw (Z-axis rotation)
    SCALAR_T siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
    SCALAR_T cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
    xyz(2) = std::atan2(siny_cosp, cosy_cosp);

  return xyz;
}

#endif /* F7E9056B_8FE2_4437_89C8_E07473EA4A18 */
