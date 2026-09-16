// Current encoder hold only. No ROS, homing, offsets, calibration files or policies.
#include "core.hpp"
#include "rt/rt_spi.h"
#include <chrono>
#include <csignal>
#include <cerrno>
#include <linux/joystick.h>
#include <sys/file.h>
#include <poll.h>
#include <thread>

using Clock=std::chrono::steady_clock;
using measurement_hold::Pose;
volatile std::sig_atomic_t stopping=0;
void stop_signal(int){stopping=1;}

struct Pad {
 int fd=-1;bool ps=false,seen=false;
 Pad(){
  fd=open("/dev/input/js0",O_RDONLY|O_NONBLOCK);
  if(fd<0)throw std::runtime_error("Connect the gamepad before hold");
  unsigned char buttons=0;char name[128]{};
  if(ioctl(fd,JSIOCGBUTTONS,&buttons)<0 || buttons<=10 || ioctl(fd,JSIOCGNAME(sizeof(name)),name)<0 ||
     std::string(name).find("Wireless Controller")==std::string::npos)
   throw std::runtime_error("Expected the tested DualSense Wireless Controller on js0");
 }
 ~Pad(){if(fd>=0)close(fd);}
 bool ok(){
  pollfd p{fd,POLLIN,0};if(poll(&p,1,0)<0 || p.revents&(POLLHUP|POLLERR|POLLNVAL))return false;
  js_event ev{};ssize_t n;
  while((n=::read(fd,&ev,sizeof(ev)))==sizeof(ev)){
   if((ev.type&~JS_EVENT_INIT)==JS_EVENT_BUTTON && ev.number==10){ps=ps||ev.value!=0;seen=true;}
  }
  return n<0 && errno==EAGAIN && seen && !ps;
 }
};

void feedback(Pose& q,Pose& v){
 auto* d=get_spi_data();
 for(int k=0;k<4;++k){
  q[3*k]=d->q_abad[k];q[3*k+1]=d->q_hip[k];q[3*k+2]=d->q_knee[k];
  v[3*k]=d->qd_abad[k];v[3*k+1]=d->qd_hip[k];v[3*k+2]=d->qd_knee[k];
 }
}
void command(const Pose& target,double gain){
 auto* c=get_spi_command();*c={};
 for(int k=0;k<4;++k){
  // Feedback and commands share the SPI driver's fixed sign convention.
  // Copy corresponding channels directly; do not wrap or apply model limits.
  c->q_des_abad[k]=target[3*k];c->q_des_hip[k]=target[3*k+1];c->q_des_knee[k]=target[3*k+2];
  c->kp_abad[k]=c->kp_hip[k]=c->kp_knee[k]=7.5*gain;
  c->kd_abad[k]=c->kd_hip[k]=c->kd_knee[k]=.35*gain;
  c->flags[k]=1;
 }
}
struct Disable {
 ~Disable(){
  // Run while the ownership lock is still held, on stop and every exception.
  *get_spi_command()={};
  for(int i=0;i<5;++i){spi_driver_run();std::this_thread::sleep_for(std::chrono::milliseconds(2));}
  std::cout<<"RELEASED: zero gains and motor enable flags\n"<<std::flush;
 }
};
struct Owner {
 int fd;
 Owner(){fd=open("/tmp/control_board_hardware.lock",O_CREAT|O_RDWR,0666);
  if(fd<0 || flock(fd,LOCK_EX|LOCK_NB)<0)throw std::runtime_error("Another hardware owner is running");}
 ~Owner(){close(fd);}
};
int main(int argc,char** argv){
 if(argc!=2 || std::string(argv[1])!="--hold"){
  std::cerr<<"Usage: current_pose_hold --hold (ramps stiffness at the measured pose; PS releases)\n";return 2;
 }
 std::signal(SIGINT,stop_signal);std::signal(SIGTERM,stop_signal);std::signal(SIGHUP,stop_signal);
 try{
  Owner owner;Pad pad;
  if(!pad.ok())throw std::runtime_error("Release PS; fresh gamepad button state required");
  init_spi();Disable disable;
  Pose q{},v{},low{},high{};unsigned samples=0;
  const auto deadline=Clock::now()+std::chrono::seconds(15);auto first=Clock::now();
  std::cout<<"Reading stationary current encoder positions with motors disabled. No calibration.\n"<<std::flush;
  bool captured=false;
  while(!stopping && Clock::now()<deadline){
   if(!pad.ok())throw std::runtime_error("PS stop or gamepad disconnected");
   spi_driver_run();feedback(q,v);
   bool valid=spi_feedback_valid();
   for(int i=0;i<12;++i)valid=valid&&std::isfinite(q[i])&&std::isfinite(v[i])&&std::abs(v[i])<=.1;
   if(!valid){samples=0;}else{
    if(!samples){low=high=q;first=Clock::now();}
    bool steady=true;
    for(int i=0;i<12;++i){low[i]=std::min(low[i],q[i]);high[i]=std::max(high[i],q[i]);steady=steady&&high[i]-low[i]<=.002;}
    if(!steady)samples=0;
    else if(++samples>=50 && Clock::now()-first>=std::chrono::seconds(1)){captured=true;break;}
   }
   std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if(!captured || stopping)throw std::runtime_error("No stationary current-pose capture; hold not enabled");
  measurement_hold::Core hold;hold.reset(q,v);
  std::cout<<"Fixed targets (SPI channel order):";for(double x:q)std::cout<<' '<<x;
  std::cout<<"\nRAMPING: kp=7.5 kd=0.35 over 3 seconds; PS releases\n"<<std::flush;
  auto previous=Clock::now();auto next=previous;bool announced=false;
  // First enabled packet has exact measured targets and zero gains.
  command(hold.target,0);spi_driver_run();
  while(!stopping){
   next+=std::chrono::microseconds(2000);std::this_thread::sleep_until(next);
   auto now=Clock::now();double dt=std::chrono::duration<double>(now-previous).count();previous=now;
   bool pad_ok=pad.ok();feedback(q,v);
   hold.step(dt,q,v,pad_ok&&spi_feedback_valid());
   if(hold.stopped)throw std::runtime_error("Hold stopped: PS/gamepad, feedback, drift, speed, torque or timing guard");
   command(hold.target,hold.gain);spi_driver_run();
   if(!announced&&hold.gain==1){announced=true;std::cout<<"HOLDING all 12 motors at fixed captured positions\n"<<std::flush;}
  }
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
