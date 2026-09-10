#include "robot_calibration/calibration.hpp"
#include <iostream>

using namespace robot_calibration;
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
template <typename F> void rejects(F f) { bool failed=false; try { f(); } catch (const std::exception &) { failed=true; } require(failed,"expected rejection"); }
Tree fixture(const std::string &session) {
  Tree tree, names, q, home, base;
  tree.put("schema_version",1); tree.put("calibration_id","test-calibration");
  tree.put("encoder_session_id",session); tree.put("operator_confirmed",true);
  tree.put("angle_units","radians"); tree.put("reference_convention","marked_point_ring_home");
  for (const auto *name : joint_names) { Tree v; v.put("",name); names.push_back({"",v}); }
  for (int i=0;i<12;++i) { Tree v; v.put("",0.); q.push_back({"",v}); }
  for (int i=0;i<4;++i) {
    Tree v; v.put("",0.); home.push_back({"",v});
    v.put("",std::acos(-1.0)); base.push_back({"",v});
  }
  tree.add_child("joint_names",names); tree.add_child("reference_joint_positions",q);
  tree.add_child("wheel_home",home); tree.add_child("wheel_base_target",base);
  return tree;
}
int main() {
  const auto tmp=fs::temp_directory_path()/ ("quadmorph-calibration-test-"+identity());
  ::setenv("QUADMORPH_CALIBRATION_DIR",tmp.c_str(),1);
  try {
    CaptureLock lock;
    rejects([] { CaptureLock competing; });
    const auto first=begin_session();
    rejects([] { current_session(); });
    finish_session(first);
    require(current_session()==first,"live session accepted");
    atomic_json(directory()/"calibration.json",fixture(first));
    require(load_current().session_id==first,"saved home loads");
    auto bad=fixture(first); bad.put("angle_units","degrees");
    atomic_json(directory()/"calibration.json",bad); rejects([] { load_current(); });
    atomic_json(directory()/"calibration.json",fixture(first));
    const auto second=begin_session(); finish_session(second);
    rejects([] { load_current(); });
    invalidate_session(first); require(current_session()==second,"old owner cannot invalidate a newer session");
    atomic_json(directory()/"calibration.json",fixture(second));
    auto session=read_json(directory()/"encoder-session.json");session.put("owner_start_ticks","0");
    atomic_json(directory()/"encoder-session.json",session);rejects([] { load_current(); });
    invalidate_session(second); rejects([] { current_session(); });
    std::cout<<"PASS: session identity, pending homing, stale record, process reuse, invalidation and locking\n";
  } catch(const std::exception &e) { std::cerr<<e.what()<<'\n';fs::remove_all(tmp);return 1; }
  fs::remove_all(tmp);return 0;
}
