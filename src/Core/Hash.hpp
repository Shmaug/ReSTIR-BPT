#pragma once

#include <vulkan/vulkan_hash.hpp>

namespace ptvk {

template<typename T> concept hashable = requires(T v) { { std::hash<T>()(v) } -> std::convertible_to<size_t>; };

constexpr size_t HashCombine(const size_t x, const size_t y) {
	return x ^ (y + 0x9e3779b9 + (x << 6) + (x >> 2));
}

// accepts string literals
template<typename T, size_t N>
constexpr size_t HashArray(const T(& arr)[N]) {
	std::hash<T> hasher;
	if constexpr (N == 0)
		return 0;
	else if constexpr (N == 1)
		return hasher(arr[0]);
	else
		return HashCombine(HashArray<T,N-1>(arr), hasher(arr[N-1]));
}

template<std::ranges::range R> requires(hashable<std::ranges::range_value_t<R>>)
inline size_t HashRange(const R& r) {
	std::hash<std::ranges::range_value_t<R>> hasher;
	size_t h = 0;
	for (auto it = std::ranges::begin(r); it != std::ranges::end(r); ++it)
		h = HashCombine(h, hasher(*it));
	return h;
}

template<hashable Tx, hashable... Ty>
inline size_t HashArgs(const Tx& x, const Ty&... y) {
	std::hash<Tx> hasher;
	if constexpr (sizeof...(Ty) == 0)
		return hasher(x);
	else
		return HashCombine(hasher(x), HashArgs<Ty...>(y...));
}


template<hashable T0, hashable T1>
struct PairHash {
	inline size_t operator()(const std::pair<T0,T1>& v) const {
		return HashArgs(v.first, v.second);
	}
};

template<hashable... Types>
struct TupleHash {
	inline size_t operator()(const std::tuple<Types...>& v) const {
		return HashArgs<Types...>(std::get<Types>(v)...);
	}
};

template<hashable T, size_t N>
struct ArrayHash {
	constexpr size_t operator()(const std::array<T,N>& a) const {
		return HashArray<T,N>(a.data());
	}
};

template<std::ranges::range R> requires(hashable<std::ranges::range_value_t<R>>)
struct RangeHash {
	constexpr size_t operator()(const R& r) const {
		return HashRange<R>(r);
	}
};

}