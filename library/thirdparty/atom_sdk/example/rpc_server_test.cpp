#include "atom/rpc/algs_rpc_server.h"

int main()
{
    Atom::AlgsRpcServer server;
    server.Init();
    // SwitchUpperLimbControl callback
    server.RegisterSetTeleSwitchCallback([&](bool is_on) {
        server.SetTeleSwitch(is_on);
        return 1;
    });
    server.Start(51234);

    while (true) {
        std::cout << "=======" << server.GetTeleSwitch() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
