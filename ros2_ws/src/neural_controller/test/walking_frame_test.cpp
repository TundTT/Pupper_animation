#include "neural_controller/walking_frame.hpp"
#include <iostream>
#include <random>
void require(bool ok){if(!ok)throw std::runtime_error("walking frame check failed");}
int main(){
  std::array<double,12> home{1,0,-1,-1,0,1,1,0,-1,-1,0,1},ref{};
  ref[2]=-.046;ref[5]=-.049;ref[8]=-.045;ref[11]=-.033;
  auto q=home;for(int i=0;i<12;++i)q[i]+=ref[i];q[11]+=2*3.141592653589793;
  const auto fixed=neural_controller::walking_offsets(q,home,ref);
  require(std::abs(fixed[11]-6.250185307179586)<1e-12);
  std::mt19937 gen(7);std::uniform_real_distribution<double> random(-1,1);
  for(int trial=0;trial<1000;++trial)for(int i=0;i<12;++i){
    const double model_q=home[i]+.2*random(gen),action=random(gen),scale=i%3==2?1.1:.25;
    require(std::abs(((model_q+fixed[i])-fixed[i]-home[i])-(model_q-home[i]))<1e-12);
    require(std::abs(((home[i]+scale*action+fixed[i])-fixed[i])-(home[i]+scale*action))<1e-12);
  }
  require(neural_controller::walking_offsets(q,home,ref)==fixed); // no cumulative reactivation offset
  q[2]+=3.141592653589793;bool rejected=false;
  try{neural_controller::walking_offsets(q,home,ref);}catch(...){rejected=true;}require(rejected);
  q=home;q[1]=.5;rejected=false;
  try{neural_controller::walking_offsets(q,home,ref);}catch(...){rejected=true;}require(rejected);
  std::cout<<"Walking full-turn mapping, observation/action equivalence and entry rejection passed\n";
}
