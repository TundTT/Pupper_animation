#include <iostream>
#include <iomanip>
#include "/mnt/c/Users/tundt/Desktop/Pupper_animation/output/heating-module-preview/alignment-geometry-for-commit.hpp"
int main(){std::array<double,12> q;std::array<double,3> g;
 std::cout<<std::setprecision(17);
 while(std::cin>>q[0]){for(int i=1;i<12;++i)std::cin>>q[i];for(auto& x:g)std::cin>>x;
 for(int k=0;k<4;++k){auto m=neural_controller::WheelAlignMotion::margins(q,g,k);for(auto x:m)std::cout<<x<<' ';}
 for(int k=0;k<4;++k){auto m=keyframe_align::Geometry::margins(q,g,k);for(auto x:m)std::cout<<x<<' ';}std::cout<<'\n';}}
