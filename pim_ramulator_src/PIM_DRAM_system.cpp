#include "memory_system/memory_system.h"
#include "translation/translation.h"
#include "dram_controller/controller.h"
#include "addr_mapper/addr_mapper.h"
#include "dram/dram.h"

namespace Ramulator {

class PIMDRAMSystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, PIMDRAMSystem, "PIMDRAM", "A PIM DRAM-based memory system.");

  protected:
    Clk_t m_clk = 0;
    IDRAM*  m_dram;
    IAddrMapper*  m_addr_mapper;
    std::vector<IDRAMController*> m_controllers;

  public:
    int s_num_read_requests = 0;
    int s_num_write_requests = 0;
    int s_num_pim_mac_all_bank_requests = 0;
    int s_num_pim_mac_same_bank_requests = 0;
    int s_num_pim_mac_per_bank_requests = 0;
    int s_num_pim_write_to_gemv_buffer_requests = 0;
    int s_num_pim_move_to_softmax_buffer_requests = 0;
    int s_num_pim_move_to_gemv_buffer_requests = 0;
    int s_num_pim_softmax_requests = 0;
    int s_num_pim_set_model_requests = 0;
    int s_num_pim_set_head_requests = 0;
    int s_num_other_requests = 0;


  public:
    void init() override { 
      // Create device (a top-level node wrapping all channel nodes)
      m_dram = create_child_ifce<IDRAM>();
      m_addr_mapper = create_child_ifce<IAddrMapper>();

      int num_channels = m_dram->get_level_size("channel");   

      // Create memory controllers
      for (int i = 0; i < num_channels; i++) {
        IDRAMController* controller = create_child_ifce<IDRAMController>();
        controller->m_impl->set_id(fmt::format("Channel {}", i));
        controller->m_channel_id = i;
        m_controllers.push_back(controller);
      }

      m_clock_ratio = param<uint>("clock_ratio").required();

      register_stat(m_clk).name("memory_system_cycles");
      register_stat(s_num_read_requests).name("total_num_read_requests");
      register_stat(s_num_write_requests).name("total_num_write_requests");
      register_stat(s_num_pim_mac_all_bank_requests).name("total_num_pim_mac_all_bank_requests");
      register_stat(s_num_pim_mac_same_bank_requests).name("total_num_pim_mac_same_bank_requests");
      register_stat(s_num_pim_mac_per_bank_requests).name("total_num_pim_mac_per_bank_requests");
      register_stat(s_num_pim_write_to_gemv_buffer_requests).name("total_num_pim_write_to_gemv_buffer_requests");
      register_stat(s_num_pim_move_to_softmax_buffer_requests).name("total_num_pim_move_to_softmax_buffer_requests");
      register_stat(s_num_pim_move_to_gemv_buffer_requests).name("total_num_pim_move_to_gemv_buffer_requests");
      register_stat(s_num_pim_softmax_requests).name("total_num_pim_softmax_requests");
      register_stat(s_num_pim_set_model_requests).name("total_num_pim_set_model_requests");
      register_stat(s_num_pim_set_head_requests).name("total_num_pim_set_head_requests");
      register_stat(s_num_other_requests).name("total_num_other_requests");
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override { }

    bool send(Request req) override {
      m_addr_mapper->apply(req);
      int channel_id = req.addr_vec[0];
      bool is_success = m_controllers[channel_id]->send(req);

      if (is_success) {
        switch (req.type_id) {
          case Request::Type::Read: {
            s_num_read_requests++;
            break;
          }
          case Request::Type::Write: {
            s_num_write_requests++;
            break;
          }
          case Request::Type::PIM_MAC_AB: {
            s_num_pim_mac_all_bank_requests++;
            break;
          }
          case Request::Type::PIM_MAC_SB: {
            s_num_pim_mac_same_bank_requests++;
            break;
          }
          case Request::Type::PIM_MAC_PB: {
            s_num_pim_mac_per_bank_requests++;
            break;
          }
          case Request::Type::PIM_WR_GB: {
            s_num_pim_write_to_gemv_buffer_requests++;
            break;
          }
          case Request::Type::PIM_MV_SB: {
            s_num_pim_move_to_softmax_buffer_requests++;
            break;
          }
          case Request::Type::PIM_MV_GB: {
            s_num_pim_move_to_gemv_buffer_requests++;
            break;
          }
          case Request::Type::PIM_SFM: {
            s_num_pim_softmax_requests++;
            break;
          }
          case Request::Type::PIM_SET_MODEL: {
            s_num_pim_set_model_requests++;
            break;
          }
          case Request::Type::PIM_SET_HEAD: {
            s_num_pim_set_head_requests++;
            break;
          }
          case Request::Type::PIM_BARRIER: {
            break;
          }
          default: {
            s_num_other_requests++;
            break;
          }
        }
      }

      return is_success;
    };
    
    void tick() override {
      m_clk++;
      m_dram->tick();
      for (auto controller : m_controllers) {
        controller->tick();
      }
    };

    float get_tCK() override {
      return m_dram->m_timing_vals("tCK_ps") / 1000.0f;
    };

    // const SpecDef& get_supported_requests() override {
    //   return m_dram->m_requests;
    // };

    bool is_pending() override {
      bool is_pending = false;
      for (auto controller : m_controllers) {
        is_pending |= controller->is_pending();
      }
      return is_pending;
    };
};
  
}   // namespace 

