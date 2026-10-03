#pragma once

#include <memory>
#include <utility>

// Move-only public wrappers keep their existing unique/shared ownership and
// per-class friends. No base class, virtual dispatch, or additional state.
// Define special members where Impl is complete, including the empty constructor:
// inline unique_ptr construction may instantiate exception cleanup of opaque Impl.
// The expansion ends in public access for the wrapper's resource-specific API.
#define METALLIC_RHI_HANDLE(Type, Storage, ...) \
public: \
    Type() noexcept; \
    ~Type(); \
    Type(Type&&) noexcept; \
    Type& operator=(Type&&) noexcept; \
    Type(const Type&) = delete; \
    Type& operator=(const Type&) = delete; \
private: \
    explicit Type(std::unique_ptr<detail::Type##Impl> impl); \
    std::Storage<detail::Type##Impl> impl_; \
    __VA_ARGS__ \
public:

// CommandBuffer uses only this part: its destructor and moves also manage the
// recording/submission transaction. Never default those operations implicitly.
#define METALLIC_RHI_HANDLE_CONSTRUCTORS(Type) \
    Type::Type() noexcept = default; \
    Type::Type(std::unique_ptr<detail::Type##Impl> impl) \
        : impl_(std::move(impl)) \
    { \
    }

// Impl owns native destruction so move assignment releases the destination's
// old resource too. Shared storage retains allocations held by submitted work.
#define METALLIC_RHI_HANDLE_DEFINITIONS(Type) \
    METALLIC_RHI_HANDLE_CONSTRUCTORS(Type) \
    Type::~Type() = default; \
    Type::Type(Type&&) noexcept = default; \
    Type& Type::operator=(Type&&) noexcept = default;
