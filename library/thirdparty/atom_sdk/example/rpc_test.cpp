#include "atom/rpc/algs_rpc_client.h"
#include <iostream>
#include <thread>

int main()
{
    Atom::AlgsRpcClient rpc("127.0.0.1", 51234);
    std::string test;
    int sw = 0;
    while (true) {
        std::cin >> test;
        std::cout << rpc.SwitchUpperLimbControl(1) << std::endl;
        std::cin >> test;
        std::cout << rpc.SwitchUpperLimbControl(0) << std::endl;
    }
    // while (true) {
    //     if (sw == 0) {
    //         std::cout << "Enter: 1 = FSM, 2 = SetVel." << std::endl;
    //         std::cin >> sw;
    //         if (std::cin.fail()) {
    //             std::cin.clear();                                                    // Clear error flags
    //             std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Discard invalid input
    //             std::cout << "Invalid input. Please enter a number." << std::endl;
    //             continue;
    //         }
    //         std::cout << "Test Start..." << std::endl;
    //     }
    //     switch (sw) {
    //         case 1: {
    //             int32_t fsm_id = 0;
    //             rpc.GetFsmId(fsm_id);
    //             std::cout << "Enter number for fsm_id: ";
    //             std::cin >> fsm_id;
    //             Atom::RpcErrorCode errorCode = rpc.SetFsmId(fsm_id);
    //             if (errorCode != Atom::RpcErrorCode::SUCCESS) { std::cout << "SetFsmId errorCode: " << errorCode << std::endl; }
    //             std::this_thread::sleep_for(std::chrono::seconds(2));
    //             break;
    //         }
    //         case 2: {
    //             int32_t fsm_id = 0;
    //             rpc.GetFsmId(fsm_id);
    //             if (301 != fsm_id && 302 != fsm_id) {
    //                 std::cout << "SetVel failed. Please enter fsm 301 or 302." << std::endl;
    //                 sw = 0;
    //                 break;
    //             }
    //             const float kVel = 0.3;
    //             const float duration = 2.0;
    //             Atom::RpcErrorCode errorCode = rpc.SetVel(kVel, 0.0, 0.0, duration);
    //             if (errorCode != Atom::RpcErrorCode::SUCCESS) { std::cout << "\033[31mSetVel failed.\033[0m" << std::endl; }
    //             sw = 0;
    //             break;
    //         }
    //         default:
    //             std::cout << "not support" << std::endl;
    //             sw = 0;
    //             break;
    //     }
    // }
    return 0;
}
