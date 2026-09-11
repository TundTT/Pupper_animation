#include <RTNeural/RTNeural.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdexcept>
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: align_export_test POLICY_JSON FIXTURES_CSV");
    std::ifstream json(argv[1]);auto m=RTNeural::json_parser::parseJson<float>(json,false);
    if(!m || m->getInSize()!=83 || m->getOutSize()!=8)throw std::runtime_error("motion network shape");
    std::ifstream input(argv[2]);std::string line;int count=0;double worst=0;
    while(std::getline(input,line)) {
      std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
      std::vector<float> obs(83),expected(8);
      for(auto&x:obs)if(!(row>>x))throw std::runtime_error("missing observation");
      for(auto&x:expected)if(!(row>>x))throw std::runtime_error("missing output");
      m->forward(obs.data());
      for(int i=0;i<8;++i){double e=std::abs(double(m->getOutputs()[i])-expected[i]);if(!std::isfinite(e)||e>3e-5)throw std::runtime_error("RTNeural mismatch");worst=std::max(worst,e);}
      ++count;
    }
    if(count!=128)throw std::runtime_error("expected 128 fixtures");
    std::cout<<"PASS: RTNeural motion export, max error "<<worst<<'\n';return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
