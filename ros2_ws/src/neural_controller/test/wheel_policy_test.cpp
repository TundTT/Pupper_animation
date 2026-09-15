#include <RTNeural/RTNeural.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdexcept>
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: wheel_policy_test POLICY_JSON FIXTURES_CSV");
    std::ifstream metadata_stream(argv[1]); nlohmann::json metadata; metadata_stream >> metadata;
    if(metadata.at("behavior")!="wheel" || metadata.at("observation_history")!=4 ||
       metadata.contains("kp") || metadata.contains("kd"))
      throw std::runtime_error("wheel behavior/history/gain contract");
    for(int row=0;row<12;++row) {
      bool wheel=row%3==2;
      if(metadata.at("action_types").at(row)!=(wheel?"velocity":"position") ||
         std::abs(metadata.at("kps").at(row).get<double>()-(wheel?0.:5.))>1e-9 ||
         std::abs(metadata.at("kds").at(row).get<double>()-(wheel?.35:.25))>1e-9 ||
         std::abs(metadata.at("init_kps").at(row).get<double>()-(wheel?0.:7.5))>1e-9 ||
         std::abs(metadata.at("init_kds").at(row).get<double>()-(wheel?.1:.25))>1e-9)
        throw std::runtime_error("mixed actuator gain or type mismatch");
      if(wheel) {
        int leg=row/3;
        double sign=leg%2==0?-1.:1.;
        if(metadata.at("wheel_joint_rows").at(leg)!=row ||
           metadata.at("wheel_forward_sign").at(leg)!=sign ||
           std::abs(metadata.at("action_scale").at(row).get<double>()-
                    sign*metadata.at("wheel_velocity_normalizer").get<double>())>1e-6)
          throw std::runtime_error("wheel direction/scale mismatch");
      }
    }
    for(size_t i=0;i<metadata.at("layers").size();++i)
      if(metadata.at("layers").at(i).at("activation")!=(i+1==metadata.at("layers").size()?"tanh":"elu"))
        throw std::runtime_error("unsupported policy activation");
    std::ifstream json(argv[1]);auto m=RTNeural::json_parser::parseJson<float>(json,false);
    if(!m || m->getInSize()!=132 || m->getOutSize()!=12)throw std::runtime_error("wheel network shape");
    std::ifstream input(argv[2]);std::string line;int count=0;double worst=0;
    while(std::getline(input,line)) {
      std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
      std::vector<float> obs(132),expected(12);
      for(auto&x:obs)if(!(row>>x))throw std::runtime_error("missing observation");
      for(auto&x:expected)if(!(row>>x))throw std::runtime_error("missing output");
      m->forward(obs.data());
      for(int i=0;i<12;++i){double e=std::abs(double(m->getOutputs()[i])-expected[i]);if(!std::isfinite(e)||e>3e-5)throw std::runtime_error("RTNeural mismatch");worst=std::max(worst,e);}
      ++count;
    }
    if(count!=128)throw std::runtime_error("expected 128 fixtures");
    std::cout<<"PASS: RTNeural wheel export, max error "<<worst<<'\n';return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
