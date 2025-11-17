#pragma once
#include <string>
#include <map>
#include <vector>
#include <typeinfo>
#include <memory>
#include <stdexcept>

class Any
{
public:
    struct Placeholder {
        virtual ~Placeholder() = default;
        virtual const std::type_info &type() const = 0;
        virtual Placeholder *clone() const = 0;
    };
    Any &operator=(const Any &other)
    {
        if (this != &other) {
            delete content;
            content = other.content ? other.content->clone() : nullptr;
        }
        return *this;
    }
    template <typename T>
    struct Holder : public Placeholder {
        T held;
        Holder(const T &value) : held(value) {}
        const std::type_info &type() const override { return typeid(T); }
        Placeholder *clone() const override { return new Holder(held); }
    };

    Any() : content(nullptr) {}

    template <typename T>
    Any(const T &value) : content(new Holder<T>(value))
    {
    }

    Any(const Any &other) : content(other.content ? other.content->clone() : nullptr) {}

    ~Any() { delete content; }

    const std::type_info &type() const { return content ? content->type() : typeid(void); }

    template <typename T>
    T &cast()
    {
        if (type() != typeid(T)) throw std::bad_cast();
        return static_cast<Holder<T> *>(content)->held;
    }

    template <typename T>
    const T &cast() const
    {
        if (type() != typeid(T)) throw std::bad_cast();
        return static_cast<const Holder<T> *>(content)->held;
    }

private:
    Placeholder *content;
};

using JsonMap = std::map<std::string, Any>;
using JsonArray = std::vector<Any>;

inline void FromAny(const Any &a, bool &value) { value = a.cast<bool>(); }
inline void FromAny(const Any &a, int32_t &value) { value = a.cast<int32_t>(); }
inline void FromAny(const Any &a, std::string &value) { value = a.cast<std::string>(); }
inline void FromAny(const Any &a, JsonMap &value) { value = a.cast<JsonMap>(); }
inline void FromAny(const Any &a, float &value) { value = a.cast<float>(); }