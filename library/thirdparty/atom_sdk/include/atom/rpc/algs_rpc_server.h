#ifndef __ALGS_RPC_SERVER_H__
#define __ALGS_RPC_SERVER_H__
#include "atom/rpc/base_rpc_server.h"

namespace Atom
{
    struct VelParams {
        float vx;
        float vy;
        float vyaw;
        float duration;
        bool isInitialized;
    };

    class AlgsRpcServer : public BaseJsonRpcServer
    {
    public:
        void Init() override;

        // SwitchUpperLimbControl
        bool GetTeleSwitch() { return is_tele_on_; }
        void SetTeleSwitch(bool sw) { this->is_tele_on_ = sw; }
        using SetTeleSwitchCallback = std::function<bool(bool)>;
        void RegisterSetTeleSwitchCallback(SetTeleSwitchCallback callback) { set_tele_switch_callback_ = callback; }
        // GetFsmId
        using GetFsmIdCallback = std::function<int32_t()>;
        void RegisterGetFsmIdCallback(GetFsmIdCallback callback) { get_fsm_id_callback_ = callback; }
        // SetFsmId
        using SetFsmIdCallback = std::function<bool(int32_t)>;
        void RegisterSetFsmIdCallback(SetFsmIdCallback callback) { set_fsm_id_callback_ = callback; }
        // SetVel
        using SetVelCallback = std::function<bool()>;
        void RegisterSetVelCallback(SetVelCallback callback) { set_vel_callback_ = callback; }
        VelParams GetVelParams() { return this->vel_params_; }
        void ResetVelParams() { this->vel_params_.isInitialized = false; }

    protected:
        // SwitchUpperLimbControl
        bool is_tele_on_ = false;
        SetTeleSwitchCallback set_tele_switch_callback_;
        // GetFsmId
        GetFsmIdCallback get_fsm_id_callback_;
        // SetFsmId
        SetFsmIdCallback set_fsm_id_callback_;
        // SetVel
        SetVelCallback set_vel_callback_;
        VelParams vel_params_;
    };

}  // namespace Atom

#endif  //__ALGS_RPC_SERVER_H__
