#ifndef __BASE_RPC_SERVER_H__
#define __BASE_RPC_SERVER_H__

#include <string>
#include <functional>
#include <map>
#include <iostream>
#include <cstdint>
#include <netinet/in.h>
#include <thread>
#include <unistd.h>

#include "atom/rpc/any.h"
#include "nlohmann/json.hpp"

namespace Atom
{
    class BaseJsonRpcServer
    {
    public:
        virtual ~BaseJsonRpcServer() = default;
        void Start(uint16_t port);
        virtual void Init() = 0;

    protected:
        using RpcHandler = std::function<void(const Any &, Any &)>;
        void RegisterMethod(const std::string &name, RpcHandler handler);

    private:
        void Run(uint16_t port);
        bool HandleRequest(const std::string &method, const JsonMap &params, Any &result);
        void HandleClient(int client_fd);
        std::map<std::string, RpcHandler> methods_;
    };

}  // namespace Atom

#endif  //__BASE_RPC_SERVER_H__
