// Exercise production packet filling/release with a fake SPI transport, no devices.
#define main hold_entry_point
#include "main.cpp"
#undef main
#include <cassert>
spi_command_t fake_command{};spi_data_t fake_data{};int transfers=0;
void init_spi(){} void spi_driver_run(){++transfers;}
bool spi_feedback_valid(){return true;}
spi_command_t* get_spi_command(){return &fake_command;}
spi_data_t* get_spi_data(){return &fake_data;}
int main(){
 Pose q{},v{};for(int k=0;k<4;++k){
  fake_data.q_abad[k]=k+.125;fake_data.q_hip[k]=-k-.25;fake_data.q_knee[k]=k*12+.5;
 }
 feedback(q,v);command(q,0);
 for(int k=0;k<4;++k){
  assert(fake_command.q_des_abad[k]==fake_data.q_abad[k]);
  assert(fake_command.q_des_hip[k]==fake_data.q_hip[k]);
  assert(fake_command.q_des_knee[k]==fake_data.q_knee[k]);
  assert(fake_command.kp_abad[k]==0&&fake_command.kd_knee[k]==0);
 }
 command(q,1);
 for(int k=0;k<4;++k){
  assert(fake_command.kp_abad[k]==7.5&&fake_command.kp_hip[k]==7.5&&fake_command.kp_knee[k]==7.5);
  assert(fake_command.qd_des_abad[k]==0&&fake_command.qd_des_hip[k]==0&&fake_command.qd_des_knee[k]==0);
  assert(fake_command.tau_abad_ff[k]==0&&fake_command.tau_hip_ff[k]==0&&fake_command.tau_knee_ff[k]==0);
  assert(fake_command.flags[k]==1);
 }
 {Disable release;}
 const spi_command_t zero{};assert(memcmp(&fake_command,&zero,sizeof(zero))==0&&transfers==5);
 std::cout<<"PASS: 12 channel roundtrip, winding preserved, zero-gain first packet, shutdown disables all motors\n";
}
