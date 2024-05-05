#include <vector>

#include "base/base.h"
#include "dram_controller/controller.h"
#include "dram_controller/scheduler.h"

namespace Ramulator {

class PIM : public IScheduler, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IScheduler, PIM, "PIM", "PIM DRAM Scheduler.")
  private:
    IDRAM* m_dram;

  public:
    void init() override { };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      m_dram = cast_parent<IDRAMController>()->m_dram;
    };

    std::vector<AddrVec_t> rowhit_list;
 
    AddrVec_t get_bank_addr_vec (Request req) {
      AddrVec_t bank_addr_vec;
      for (auto itr = req.addr_vec.begin(); itr != (req.addr_vec.begin() + m_dram->m_levels("row")); itr++) {
        bank_addr_vec.push_back(*itr);
      }
      return bank_addr_vec;
    };


    ReqBuffer::iterator compare(ReqBuffer::iterator req1, ReqBuffer::iterator req2) override {
      bool ready1 = m_dram->check_ready(req1->command, req1->addr_vec);
      bool ready2 = m_dram->check_ready(req2->command, req2->addr_vec);

      if (ready1 ^ ready2) {
        if (ready1) {
          return req1;
        }
        else {
          if (!m_dram->check_rowbuffer_hit(req2->command, req2->addr_vec)) {
            if (std::find(rowhit_list.begin(), rowhit_list.end(), get_bank_addr_vec(*req2)) != rowhit_list.end()) {          
              return req1;
            }
          }
          return req2;
        }
      }

      // Fallback to FCFS
      if (req1->arrive <= req2->arrive) {
        return req1;
      } else {
        return req2;
      } 
    }

    ReqBuffer::iterator get_best_request(ReqBuffer& buffer) override {
      if (buffer.size() == 0) {
        return buffer.end();
      }

      for (auto& req : buffer) {
        req.command = m_dram->get_preq_command(req.final_command, req.addr_vec);
      }

      // Store row buffer hit history
      rowhit_list.clear();
      for (auto& req : buffer) {
        if (m_dram->check_rowbuffer_hit(req.command, req.addr_vec)) {
          rowhit_list.push_back(get_bank_addr_vec(req));
        }
      }

      auto candidate = buffer.begin();

      if (candidate->type_id == Request::Type::PIM_BARRIER) {
        buffer.remove(candidate);
        if (buffer.size() == 0) {
          return buffer.end();
        }
        candidate = buffer.begin();
      }

      bool barrier = false;
      for (auto next = std::next(buffer.begin(), 1); next != buffer.end(); next++) {
        if (next->type_id == Request::Type::PIM_BARRIER) {
          barrier = true;
        }
        if (barrier == false || m_dram->m_command_meta(next->command).is_opening || m_dram->m_command_meta(next->command).is_closing) {
          candidate = compare(candidate, next);
        }
      }
      return candidate;
    }
};

}       // namespace Ramulator
