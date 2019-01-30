#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// class AliasTracker
//
// This class tracks the "A points to B" graph for all values, as well as
// wildcards and writes. It is used by AliasDb to provide a higher-level API.
class AliasTracker {
 public:
  // Returns true iff `v` is present in the alias set tracker.
  bool contains(const Value* v) const;

  // Do `a` and `b` potentially share a memory location?
  bool mayAlias(const Value* a, const Value* b) const;

  // Do any values in group `a` potentailly share a memory location with any
  // value in group `b`?
  //
  // This is written so that either of the inputs could be a multiset
  template <typename T, typename U>
  bool mayAlias(const T& a, const U& b) const {
    if (a.empty() || b.empty()) {
      return false;
    }

    // Record all memory locations from group `a`
    std::unordered_set<const Element*> memoryLocations;
    for (auto it = a.cbegin(); it != a.cend();) {
      const auto value = *it;
      if (isWildcard(value)) {
        return true;
      }

      if (map_.count(value)) {
        for (const auto loc : map_.at(value)->getMemoryLocations()) {
          memoryLocations.insert(loc);
        }
      }

      const auto cnt = a.count(*it);
      std::advance(it, cnt);
    }

    // If any of group `b`s memory locations overlap, return true.
    for (auto it = b.cbegin(); it != b.cend();) {
      const auto value = *it;
      if (isWildcard(value)) {
        return true;
      }

      if (map_.count(value)) {
        for (const auto loc : map_.at(value)->getMemoryLocations()) {
          if (memoryLocations.count(loc)) {
            return true;
          }
        }
      }

      const auto cnt = b.count(*it);
      std::advance(it, cnt);
    }
    // No overlap, so group `a` and `b` do not share a memory location
    return false;
  }

  // Does `n` write to `v` directly? (Does not consider aliases)
  bool writesTo(Node* n, const Value* v) const;

  // Make `v` point at `to`.
  void makePointerTo(const Value* v, const Value* to);

  // Give `v` a fresh alias (i.e. it does not point to any value)
  void makeFreshValue(const Value* v);

  // Register `v` as a wildcard value.
  void setWildcard(const Value* v);

  // is `v` a wildcard?
  bool isWildcard(const Value* v) const;

  // Register the fact that `n` writes to `v`.
  void registerWrite(const Value* v, Node* n);

  // Return all aliases of `v`. This is the full set of any other value that
  // *may* represent the same memory location.
  // NOTE: this does not consider wildcard values
  std::unordered_set<const Value*> getAliases(const Value* v) const;

  // Does anything write to the memory locations that `v` may point to?
  bool hasWriters(const Value* v) const;

  // Get all nodes that write to a wildcard value.
  const std::unordered_set<Node*>& getWildcardWriters() const {
    return wildcardWriters_;
  }

  void dump() const;

 private:
  enum class BfsDirection {
    POINTS_TO,
    POINTED_FROM,
    // Consider both pointer directions. The closure obtained from this
    // represents the whole "alias set" of a value.
    BOTH
  };
  // `Element` represents the vertex in the points-to graph. It has a 1:1
  // relationship with IR `Value`s.
  struct Element {
    const Value* value = nullptr;
    // All values that this value *may* point to. It's possible to have multiple
    // values that you might point to due to control flow/complex ops
    std::unordered_set<Element*> pointsTo;
    // Backreference to values that point to `this`
    std::unordered_set<Element*> pointedFrom;

    std::unordered_set<const Element*> getMemoryLocations() const;
    mutable std::unordered_set<const Element*> cachedMemoryLocations_;

    // Do a breadth-first search over the graph, starting at `this` and
    // traversing in the direction `dir`.`fn` will be run on each element.
    //
    // If `shortCircuit` is set, then if `fn` evaluates to true the search will
    // short-circuit and return true. You can use this to do existence checks
    // on the graph or whatever.
    template <typename Fn>
    bool bfs(Fn fn, BfsDirection dir, bool shortCircuit = false) const;
  };

  // Structure that owns all the element pointers. It's a map of
  //  raw pointer -> unique_ptr to facilitate easy queries
  std::unordered_map<Element*, std::unique_ptr<Element>> elements_;
  // Index to look up whatever element corresponds to that value.
  std::unordered_map<const Value*, Element*> map_;
  // All values that may point to a wildcard value.
  std::unordered_set<const Value*> wildcards_;
  // All nodes that write to a wildcard
  std::unordered_set<Node*> wildcardWriters_;
  size_t numWrites_ = 0;

  std::unordered_map<Node*, std::unordered_set<const Value*>> writeIndex_;
  mutable std::unordered_set<const Element*> cachedWrittenToLocs_;
  mutable bool cacheStale_ = true;
  void rebuildCache() const;
};

} // namespace jit
} // namespace torch
