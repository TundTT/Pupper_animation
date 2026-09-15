#include <sys/ioctl.h>
#include "rt/rt_spi.h"
#include <cstdarg>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <pthread.h>

extern pthread_mutex_t spi_mutex;
extern int spi_1_fd, spi_2_fd;
static int mode = 0;
extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
  if (request != SPI_IOC_MESSAGE(1)) throw std::runtime_error("Unexpected device operation");
  va_list args; va_start(args, request);
  auto* transfer = va_arg(args, spi_ioc_transfer*);
  va_end(args);
  if (mode == 1 && fd == spi_2_fd) return -1;
  if (mode == 2) return 8;  // Short transfer.
  spine_data_t packet{};
  if (mode != 3) {
    for (int i = 0; i < 2; ++i) {
      packet.q_abad[i] = 1.f; packet.q_hip[i] = 2.f; packet.q_knee[i] = 15.f;
    }
  }
  if (mode == 5) packet.q_knee[0] = std::numeric_limits<float>::quiet_NaN();
  uint32_t checksum = 0;
  // memcpy avoids type-punning UB in the fixture itself.
  uint32_t words[15]; std::memcpy(words, &packet, sizeof(packet));
  for (int i = 0; i < 14; ++i) checksum ^= words[i];
  packet.checksum = checksum ^ (mode == 4 ? 1 : 0);
  uint16_t halves[30]; std::memcpy(halves, &packet, sizeof(packet));
  auto* rx = reinterpret_cast<uint16_t*>(transfer->rx_buf);
  for (int i = 0; i < 30; ++i) rx[i] = (halves[i] >> 8) | (halves[i] << 8);
  return 132;
}

int main() {
  // Never call init_spi/open; descriptors are sent only to the wrapped ioctl above.
  spi_1_fd = 100; spi_2_fd = 101; pthread_mutex_init(&spi_mutex, nullptr);
  spi_driver_run();
  if (!spi_feedback_valid()) throw std::runtime_error("Valid response rejected");
  for (mode = 1; mode <= 5; ++mode) {
    spi_driver_run();
    if (spi_feedback_valid()) throw std::runtime_error("Invalid response accepted");
  }
  mode = 0; spi_driver_run();
  if (!spi_feedback_valid()) throw std::runtime_error("Fresh recovery rejected");
  pthread_mutex_destroy(&spi_mutex);
  std::cout << "PASS SPI short/failed/empty/corrupt/nonfinite response rejection\n";
}
