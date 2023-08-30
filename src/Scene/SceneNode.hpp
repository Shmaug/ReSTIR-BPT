#pragma once

#include <typeindex>

#include <Common/Common.h>
#include <Core/Utils.hpp>

namespace ptvk {

// Scene graph node. Nodes hold pointers to parent/children nodes, as well as Components (just std::shared_ptr's of arbitrary types).
class SceneNode : public std::enable_shared_from_this<SceneNode> {
private:
	std::string mName;
	std::unordered_map<std::type_index, std::shared_ptr<void>> mComponents;

	std::weak_ptr<SceneNode> mParent;
	std::unordered_set<std::shared_ptr<SceneNode>> mChildren;

	SceneNode(const std::string& name) : mName(name) {}

public:
	[[nodiscard]] inline static std::shared_ptr<SceneNode> Create(const std::string& name) {
		return std::shared_ptr<SceneNode>(new SceneNode(name));
	}

	SceneNode(SceneNode&&) = default;
	SceneNode(const SceneNode&) = delete;
	SceneNode& operator=(SceneNode&&) = default;
	SceneNode& operator=(const SceneNode&) = delete;

	inline const std::string& GetName() const { return mName; }
	inline std::shared_ptr<SceneNode> GetPtr() { return shared_from_this(); }

	// Parent/child functions

	inline std::shared_ptr<SceneNode> GetParent() const { return mParent.lock(); }
	inline const auto& GetChildren() const { return mChildren; }

	inline std::shared_ptr<SceneNode> GetRoot() {
		std::shared_ptr<SceneNode> n = GetPtr();
		while (std::shared_ptr<SceneNode> p = n->GetParent()) {
			n = p;
		}
		return n;
	}

	inline void AddChild(const std::shared_ptr<SceneNode>& c) {
		if (!c) return;
		c->RemoveParent();
		c->mParent = GetPtr();
		mChildren.emplace(c);
	}
	inline std::shared_ptr<SceneNode> AddChild(const std::string& name) {
		const std::shared_ptr<SceneNode> c = Create(name);
		AddChild(c);
		return c;
	}
	inline void RemoveChild(const std::shared_ptr<SceneNode>& c) {
		if (auto it = mChildren.find(c); it != mChildren.end()) {
			mChildren.erase(it);
			c->mParent.reset();
		}
	}
	inline void RemoveParent() {
		if (auto p = GetParent()) {
			p->RemoveChild(GetPtr());
		}
	}

	// Components

	inline bool HasComponent(const std::type_index type) const { return mComponents.find(type) != mComponents.end(); }
	template<typename T>
	inline bool HasComponent() const { return HasComponent(typeid(T)); }

	template<typename T>
	inline std::shared_ptr<T> GetComponent() const {
		auto it = mComponents.find(typeid(T));
		if (it == mComponents.end())
			return nullptr;
		return static_pointer_cast<T>(it->second);
	}
	inline std::shared_ptr<void> GetComponent(const std::type_index type) const {
		auto it = mComponents.find(type);
		if (it == mComponents.end())
			return nullptr;
		return it->second;
	}
	inline auto GetComponents() const { return mComponents | std::views::keys; }

	inline void AddComponent(const std::type_index type, const std::shared_ptr<void>& v) {
		mComponents.emplace(type, v);
	}
	template<typename T>
	inline void AddComponent(const std::shared_ptr<T>& v) {
		AddComponent(typeid(T), v);
	}

	template<typename T>
	inline void RemoveComponent() {
		auto it = mComponents.find(typeid(T));
		if (it != mComponents.end()) {
			it->second.reset();
			mComponents.erase(it);
		}
	}
	inline void RemoveComponent(const std::type_index type) {
		mComponents.erase(type);
	}

	template<typename T, typename...Types>
	inline std::shared_ptr<T> MakeComponent(Types&&... args) {
		std::shared_ptr<T> c = std::make_shared<T>( std::forward<Types>(args)... );
		AddComponent(c);
		return c;
	}

	// forEach

	template<std::invocable<SceneNode&> F>
	inline void ForEachDescendant(F&& fn) {
		std::stack<std::shared_ptr<SceneNode>> todo;
		todo.push(GetPtr());
		while (!todo.empty()) {
			const std::shared_ptr<SceneNode> n = todo.top();
			todo.pop();
			fn(*n);
			for (const std::shared_ptr<SceneNode>& c : n->mChildren)
				todo.push(c);
		}
	}
	template<std::invocable<SceneNode&> F>
	inline void ForEachAncestor(F&& fn) {
		std::shared_ptr<SceneNode> n = GetPtr();
		while (n) {
			fn(*n);
			n = n->GetParent();
		}
	}

	// forEach component

	template<typename T, std::invocable<SceneNode&, std::shared_ptr<T>> F>
	inline void ForEachDescendant(F&& fn) {
		ForEachDescendant([&](SceneNode& n) {
			if (const std::shared_ptr<T> c = n.GetComponent<T>())
				fn(n, c);
		});
	}

	template<typename T, std::invocable<SceneNode&, std::shared_ptr<T>> F>
	inline void ForEachAncestor(F&& fn) {
		ForEachAncestor([&](SceneNode& n) {
			if (const std::shared_ptr<T> c = n.GetComponent<T>())
				fn(n, c);
		});
	}

	// find (stops when fn evaluates to false)

	template<typename F> requires(std::is_invocable_r_v<bool, F, SceneNode&>)
	inline void FindDescendant(F&& fn) {
		std::stack<std::shared_ptr<SceneNode>> todo;
		todo.push(GetPtr());
		while (!todo.empty()) {
			const std::shared_ptr<SceneNode> n = todo.top();
			todo.pop();
			if (!fn(*n)) break;
			for (const std::shared_ptr<SceneNode>& c : n->mChildren)
				todo.push(c);
		}
	}
	template<typename F> requires(std::is_invocable_r_v<bool, F, SceneNode&>)
	inline void FindAncestor(F&& fn) {
		std::shared_ptr<SceneNode> n = GetPtr();
		while (n) {
			if (!fn(*n)) break;
			n = n->GetParent();
		}
	}

	// returns true if p is an ancestor of this node
	inline bool IsAncestor(SceneNode& p) {
		bool found = false;
		FindAncestor([&](SceneNode& n){
			if (&n == &p) {
				found = true;
				return false;
			}
			return true;
		});
		return found;
	}
	// returns true if p is a descendant of this node
	inline bool IsDescendant(SceneNode& p) {
		bool found = false;
		FindDescendant([&](SceneNode& n){
			if (&n == &p) {
				found = true;
				return false;
			}
			return true;
		});
		return found;
	}

	// component find functions

	template<typename T>
	inline std::shared_ptr<T> FindDescendant(std::shared_ptr<SceneNode>* oNode = nullptr) {
		std::shared_ptr<T> ptr;
		FindDescendant([&](SceneNode& n) {
			if (const std::shared_ptr<T> c = n.GetComponent<T>()) {
				if (oNode) *oNode = n.GetPtr();
				ptr = c;
				return false;
			}
			return true;
		});
		return ptr;
	}
	template<typename T>
	inline std::shared_ptr<T> FindAncestor(std::shared_ptr<SceneNode>* oNode = nullptr) {
		std::shared_ptr<T> ptr;
		FindAncestor([&](SceneNode& n) {
			if (const std::shared_ptr<T> c = n.GetComponent<T>()) {
				if (oNode) *oNode = n.GetPtr();
				ptr = c;
				return false;
			}
			return true;
		});
		return ptr;
	}
};

inline float4x4 NodeToWorld(const SceneNode& node) {
	float4x4 transform = glm::identity<float4x4>();
	if (auto m = node.GetComponent<float4x4>())
		transform = *m;
	std::shared_ptr<SceneNode> p = node.GetParent();
	while (p) {
		if (auto m = p->GetComponent<float4x4>())
			transform = (*m) * transform;
		p = p->GetParent();
	}
	return transform;
}

}